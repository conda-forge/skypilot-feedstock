#! /usr/bin/env python3

# This is an auxiliary script to keep the recipe in sync with the upstream
# PyPI metadata. It is not used in the recipe itself.
#
#   ./update.py                     report what is out of date
#   ./update.py --version 0.13.0    target a version other than the recipe's
#   ./update.py --apply             rewrite recipe.yaml in place
#
# The recipe ships one output per upstream extra. Which extras are shippable
# is *derived*, not hand-maintained: every dependency is resolved to a conda
# name and checked against conda-forge, so a new upstream extra is classified
# automatically rather than crashing the script.

from typing import Dict, List, Optional, Sequence, Tuple
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from pathlib import Path
import argparse
import difflib
import json
import requests
import logging
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
import re
import tempfile
import time
from conda_lock.lookup import pypi_name_to_conda_name, DEFAULT_MAPPING_URL
from conda_lock.src_parser.pyproject_toml import poetry_version_to_conda_version

current_dir = Path(__file__).parent
recipe_path = current_dir / "recipe.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("update")

PIN_SUBPACKAGE = "${{ pin_subpackage('skypilot', exact=True) }}"

# The 'all' extra is the union of every other extra, so shipping it would be
# both excessive and redundant.
IGNORED_EXTRAS = {"all"}

# Stands in for the package itself in the triage table, which reports the core
# output alongside the extras.
CORE = "(core)"

# The package is noarch, so markers cannot be honoured at install time. We
# evaluate them against every environment the recipe supports and keep a
# requirement if any of them needs it. Only the variables upstream actually
# branches on are varied; the rest default to the running interpreter.
NOARCH_ENVIRONMENTS = [
    {"python_version": python_version, "sys_platform": sys_platform}
    for python_version in ("3.9", "3.10", "3.11", "3.12", "3.13")
    for sys_platform in ("linux", "darwin", "win32")
]

# The grayskull PyPI->conda mapping occasionally resolves a name to a
# conda-forge package that exists but is a different project. Such cases
# cannot be caught by an existence check, so they are overridden here.
# Every entry needs a reason.
NAME_OVERRIDES = {
    # conda-forge's "kubernetes" is the cluster tooling ("Production-Grade
    # Container Orchestration"). The Python client is "python-kubernetes".
    "kubernetes": "python-kubernetes",
}

# A single listing of the whole channel answers most availability questions.
# It is cached between runs, but only briefly, so that a package landing on
# conda-forge is picked up without needing the cache to be cleared by hand.
CHANNELDATA_URL = "https://conda.anaconda.org/conda-forge/channeldata.json"
CHANNELDATA_MAX_AGE_SECONDS = 24 * 60 * 60
channeldata_cache_path = (
    Path(tempfile.gettempdir()) / "skypilot-feedstock-channeldata.json"
)


def is_required_for_extras(requirement: Requirement, extras: Sequence[str]) -> bool:
    """Whether a requirement is needed for the given extras (empty meaning core).

    The package is noarch, so a single output has to satisfy every environment
    it might be installed into. A requirement therefore counts as required if
    its marker holds for *any* supported environment. Upstream splits several
    dependencies across python_version and sys_platform markers; each resulting
    constraint is listed separately in the recipe and they intersect.
    """
    if not requirement.marker:
        # No marker means it's a core requirement
        return True
    for extra in extras or [""]:
        for environment in NOARCH_ENVIRONMENTS:
            if requirement.marker.evaluate({**environment, "extra": extra}):
                return True
    return False


def patch_upstream_requirements(requirements: Sequence[str]) -> Sequence[str]:
    """
    On conda-forge we have ray[default] provided by ray-default and
    uvicorn[standard] provided by uvicorn-standard.
    """
    result = [
        re.sub(r'^ray\[default\]', 'ray-default', req)
        for req in requirements
    ]
    result = [
        re.sub(r'^uvicorn\[standard\]', 'uvicorn-standard', req)
        for req in result
    ]
    return result


def fetch_metadata(name: str, version: str) -> dict:
    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    logger.info(f"Fetching metadata from {metadata_url}")
    response = requests.get(metadata_url)
    response.raise_for_status()
    metadata = response.json()

    if not metadata["info"]["name"] == name:
        raise ValueError(
            f"Requested name {name} does not match metadata name {metadata['info']['name']}"
        )
    if not metadata["info"]["version"] == version:
        raise ValueError(
            f"Requested version {version} does not match "
            f"metadata version {metadata['info']['version']}"
        )
    return metadata


def sdist_sha256(metadata: dict) -> str:
    """The sha256 of the sdist, which is what the recipe's source URL points at."""
    sdists = [url for url in metadata["urls"] if url["packagetype"] == "sdist"]
    if len(sdists) != 1:
        raise ValueError(f"Expected exactly one sdist, found {len(sdists)}")
    return sdists[0]["digests"]["sha256"]


def conda_name(pypi_name: str, latest: Optional[Dict[str, str]] = None) -> str:
    """The conda-forge name for a PyPI package.

    Where the mapping has no entry it falls back to the normalised PyPI name,
    which gets the punctuation wrong for feedstocks that kept an underscore
    (huggingface_hub, for one). When we know what the channel contains, try the
    punctuation variants before declaring a package missing.
    """
    if pypi_name.lower() in NAME_OVERRIDES:
        return NAME_OVERRIDES[pypi_name.lower()]
    name = pypi_name_to_conda_name(pypi_name, mapping_url=DEFAULT_MAPPING_URL)
    if latest is not None and name not in latest:
        for candidate in (name.replace("-", "_"), name.replace("_", "-")):
            if candidate in latest:
                return candidate
    return name


def conda_requirement(requirement: Requirement, latest: Dict[str, str]) -> str:
    """Render an upstream requirement as a conda-forge match spec.

    Any marker has already been accounted for by is_required_for_extras, and a
    noarch output cannot express one anyway, so only the name and the version
    specifier survive.
    """
    if requirement.extras:
        raise NotImplementedError(f"Contains extras: {requirement}")
    name = conda_name(requirement.name, latest)
    version = poetry_version_to_conda_version(str(requirement.specifier))
    return f"{name} {version}".rstrip()


def latest_conda_forge_versions(session: requests.Session) -> Dict[str, str]:
    """Every conda-forge package name mapped to its newest published version.

    One request for the whole channel is far cheaper than querying each package
    individually, and it is what answers most questions we have.
    """
    if channeldata_cache_path.exists():
        age = time.time() - channeldata_cache_path.stat().st_mtime
        if age < CHANNELDATA_MAX_AGE_SECONDS:
            return json.loads(channeldata_cache_path.read_text())
    logger.info(f"Fetching {CHANNELDATA_URL}")
    response = session.get(CHANNELDATA_URL)
    response.raise_for_status()
    latest = {
        name: info.get("version")
        for name, info in response.json()["packages"].items()
    }
    channeldata_cache_path.write_text(json.dumps(latest))
    return latest


def all_conda_forge_versions(
    name: str, cache: Dict[str, List[str]], session: requests.Session
) -> List[str]:
    """Every version of a package, fetched only when the newest one is too old."""
    if name not in cache:
        response = session.get(f"https://api.anaconda.org/package/conda-forge/{name}")
        response.raise_for_status()
        cache[name] = response.json().get("versions", [])
    return cache[name]


def satisfied_by(specifier, versions: Sequence[Optional[str]]) -> bool:
    for version in versions:
        if version is None:
            continue
        try:
            if specifier.contains(Version(version), prereleases=True):
                return True
        except InvalidVersion:
            continue
    return False


def blocker(
    requirement: Requirement,
    latest: Dict[str, str],
    cache: Dict[str, List[str]],
    session: requests.Session,
) -> Optional[str]:
    """Why this requirement cannot be satisfied on conda-forge, if it cannot.

    A package can exist yet be too old to satisfy the upstream constraint, which
    is just as much a reason not to ship an output as it being absent entirely.
    """
    name = conda_name(requirement.name, latest)
    if name not in latest:
        return f"{name} (absent)"
    if not requirement.specifier:
        return None
    if satisfied_by(requirement.specifier, [latest[name]]):
        return None
    # The newest build is not acceptable, but an older one may still be.
    if satisfied_by(requirement.specifier, all_conda_forge_versions(name, cache, session)):
        return None
    return f"{name} {requirement.specifier} (conda-forge has {latest[name]})"


def split_requirements(
    metadata: dict,
) -> Tuple[List[Requirement], Dict[str, List[Requirement]]]:
    """Split the upstream requirements into core requirements and per-extra ones."""
    extras = metadata["info"]["provides_extra"]
    raw_upstream_requirements = metadata["info"]["requires_dist"]
    patched_upstream_requirements = patch_upstream_requirements(raw_upstream_requirements)
    upstream_requirements = [Requirement(req) for req in patched_upstream_requirements]

    core_requirements = [
        req for req in upstream_requirements if is_required_for_extras(req, [])
    ]
    # Every extra output depends on an exact pin of the core package, so a
    # constraint the core already imposes does not need repeating. Upstream
    # restates its server dependencies under each cloud extra; a constraint is
    # only kept when it is tighter than the core one (aws pinning colorama, say).
    core_constraints = {
        (canonicalize_name(req.name), str(req.specifier)) for req in core_requirements
    }

    def is_redundant(requirement: Requirement) -> bool:
        key = (canonicalize_name(requirement.name), str(requirement.specifier))
        return key in core_constraints

    requirements_for_extras = {
        extra: [
            req
            for req in upstream_requirements
            if req not in core_requirements
            and not is_redundant(req)
            and is_required_for_extras(req, [extra])
        ]
        for extra in extras
    }

    accounted_for = (
        set(core_requirements)
        | set(sum(requirements_for_extras.values(), []))
        | {req for req in upstream_requirements if is_redundant(req)}
    )
    if not set(upstream_requirements) == accounted_for:
        logger.error(
            "Upstream requirements do not match core requirements "
            "and requirements for extras:"
        )
        for req in accounted_for - set(upstream_requirements):
            logger.error(f"Unexpected requirement: {req}")
        for req in set(upstream_requirements) - accounted_for:
            logger.error(f"Missing requirement: {req}")
        raise ValueError("Inconsistent requirements.")

    return core_requirements, requirements_for_extras


def deduplicate(requirements: Sequence[Requirement]) -> List[Requirement]:
    """Drop repeated requirements while preserving upstream order."""
    seen = set()
    result = []
    for requirement in requirements:
        key = str(requirement)
        if key not in seen:
            seen.add(key)
            result.append(requirement)
    return result


def triage_extras(
    core_requirements: List[Requirement],
    requirements_for_extras: Dict[str, List[Requirement]],
    latest: Dict[str, str],
    session: requests.Session,
) -> Dict[str, Tuple[str, List[str]]]:
    """Classify each extra as shippable, vacuous, ignored, or blocked.

    Returns a mapping of extra -> (status, blockers), where status is one of
    "ship", "vacuous", "ignored", or "blocked".
    """
    cache: Dict[str, List[str]] = {}
    triage: Dict[str, Tuple[str, List[str]]] = {}
    core_blockers = sorted(
        {
            reason
            for reason in (
                blocker(req, latest, cache, session) for req in core_requirements
            )
            if reason is not None
        }
    )
    triage[CORE] = ("blocked" if core_blockers else "ship", core_blockers)
    for extra, requirements in sorted(requirements_for_extras.items()):
        if extra in IGNORED_EXTRAS:
            triage[extra] = ("ignored", [])
            continue
        if not requirements:
            triage[extra] = ("vacuous", [])
            continue
        blockers = sorted(
            {
                reason
                for reason in (
                    blocker(req, latest, cache, session) for req in requirements
                )
                if reason is not None
            }
        )
        triage[extra] = ("blocked" if blockers else "ship", blockers)
    return triage


def report_triage(triage: Dict[str, Tuple[str, List[str]]]) -> None:
    width = max(len(extra) for extra in triage)
    for extra, (status, blockers) in sorted(triage.items()):
        detail = {
            "ship": "all dependencies available",
            "vacuous": "no requirements of its own",
            "ignored": "union of all other extras",
            "blocked": "; ".join(blockers),
        }[status]
        print(f"  {extra:<{width}}  {status.upper():<7}  {detail}")


def diff_run_requirements(current: List[str], expected: List[str]) -> List[str]:
    """Human-readable description of how current differs from expected."""
    messages = []
    for tag, i1, i2, j1, j2 in match_by_name(current, expected):
        if tag == "equal":
            for current_entry, expected_entry in zip(current[i1:i2], expected[j1:j2]):
                if current_entry != expected_entry:
                    messages.append(f"Update '{current_entry}' to '{expected_entry}'")
        else:
            paired = min(i2 - i1, j2 - j1)
            for current_entry, expected_entry in zip(
                current[i1 : i1 + paired], expected[j1 : j1 + paired]
            ):
                messages.append(f"Update '{current_entry}' to '{expected_entry}'")
            for expected_entry in expected[j1 + paired : j2]:
                messages.append(f"Add '{expected_entry}'")
            for current_entry in current[i1 + paired : i2]:
                messages.append(f"Remove '{current_entry}'")
    return messages


def match_by_name(current: List[str], expected: List[str]):
    """Align two run lists by package name, ignoring version specifiers.

    Matching on the name alone means a changed specifier shows up as an
    in-place update rather than an unrelated removal and addition.
    """
    matcher = difflib.SequenceMatcher(
        a=[entry.split()[0] for entry in current],
        b=[entry.split()[0] for entry in expected],
        autojunk=False,
    )
    return matcher.get_opcodes()


def apply_run_requirements(run, current: List[str], expected: List[str], offset: int) -> None:
    """Edit the run list in place, disturbing as few entries as possible.

    Entries are matched by package name so that a changed version specifier is
    an in-place assignment, which lets ruamel keep the comments attached to the
    surrounding entries.
    """
    # Apply in reverse so that earlier indices stay valid.
    for tag, i1, i2, j1, j2 in reversed(match_by_name(current, expected)):
        if tag == "equal":
            for index, expected_entry in zip(range(i1, i2), expected[j1:j2]):
                if run[offset + index] != expected_entry:
                    run[offset + index] = expected_entry
            continue
        # Overwrite as many entries as line up, then insert or delete the rest,
        # so that surrounding comments survive wherever possible.
        paired = min(i2 - i1, j2 - j1)
        for index, expected_entry in zip(range(i1, i1 + paired), expected[j1 : j1 + paired]):
            run[offset + index] = expected_entry
        for index in reversed(range(i1 + paired, i2)):
            del run[offset + index]
        for expected_entry in reversed(expected[j1 + paired : j2]):
            run.insert(offset + i1 + paired, expected_entry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the recipe with PyPI metadata.")
    parser.add_argument(
        "--version",
        help="Upstream version to sync to (default: the version in the recipe)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite recipe.yaml in place instead of only reporting",
    )
    args = parser.parse_args()

    # Round-trip mode, configured to match the recipe's existing layout so that
    # rewriting it produces a reviewable diff rather than reflowing every line.
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    with open(recipe_path, "r") as f:
        recipe = yaml.load(f)

    name = recipe["recipe"]["name"]
    current_version = str(recipe["context"]["version"])
    version = args.version or current_version
    logger.info(f"Recipe {name} version: {current_version}")
    if version != current_version:
        logger.info(f"Syncing to version: {version}")

    metadata = fetch_metadata(name, version)
    extras = metadata["info"]["provides_extra"]
    core_requirements, requirements_for_extras = split_requirements(metadata)

    session = requests.Session()
    latest = latest_conda_forge_versions(session)

    print("\nOutputs:")
    triage = triage_extras(core_requirements, requirements_for_extras, latest, session)
    report_triage(triage)
    if triage[CORE][0] == "blocked":
        print(
            "\nThe core package itself cannot be built on conda-forge:\n  "
            + "\n  ".join(triage[CORE][1])
        )

    outputs = {output["package"]["name"]: output for output in recipe["outputs"]}
    assert name in outputs, f"{name} not found in outputs"
    outputs_for_extras: Dict[Optional[str], dict] = {}
    for output in outputs.keys():
        if output == name:
            outputs_for_extras[None] = outputs[output]
            continue
        if not output.startswith(f"{name}-"):
            raise ValueError(f"{output} is not an expected output for {name}.")
        output_suffix = output[len(name) + 1 :]
        if output_suffix not in extras:
            raise ValueError(
                f"{output_suffix} is no longer an upstream extra for {name}."
            )
        outputs_for_extras[output_suffix] = outputs[output]

    shippable = {
        extra
        for extra, (status, _) in triage.items()
        if status == "ship" and extra != CORE
    }
    unshipped = shippable - set(outputs_for_extras)
    if unshipped:
        # Not a defect. Outputs are published on request rather than for every
        # extra upstream adds, so this is the list to quote when someone asks.
        print(f"\nShippable on request: {', '.join(sorted(unshipped))}")
    obsolete = sorted(
        extra
        for extra in outputs_for_extras
        if extra is not None and triage.get(extra, ("missing", []))[0] != "ship"
    )
    if obsolete:
        print(f"\nOutputs whose extra is no longer shippable: {', '.join(obsolete)}")

    print()
    needs_update = False
    if version != current_version:
        needs_update = True
        print(f"Update version '{current_version}' to '{version}'")
        print("Update sha256 and reset build number to 0")

    for extra, output_recipe in outputs_for_extras.items():
        if extra is None:
            requirements = core_requirements
            print("Checking core requirements")
        else:
            requirements = requirements_for_extras[extra]
            print(f"Checking '{extra}' extra requirements")
        expected = [
            conda_requirement(req, latest) for req in deduplicate(requirements)
        ]

        run = output_recipe["requirements"]["run"]
        # The leading python entry and the trailing pin are recipe scaffolding
        # rather than upstream requirements, so they are held aside.
        offset = sum(1 for entry in run if entry.startswith("python "))
        if offset > 1:
            raise ValueError(f"Unexpected python entries in run: {list(run)}")
        trailing = 0
        if extra is not None:
            if run[-1] != PIN_SUBPACKAGE:
                raise ValueError(
                    f"Last requirement '{run[-1]}' is not a pin of the package."
                )
            trailing = 1
        current = list(run)[offset : len(run) - trailing]

        for message in diff_run_requirements(current, expected):
            print(f"  {message}")
            needs_update = True

        if args.apply:
            apply_run_requirements(run, current, expected, offset)

    if args.apply:
        if version != current_version:
            # Only a new upstream version restarts the build numbering; a
            # rebuild of the same version has to keep counting up by hand.
            recipe["context"]["version"] = DoubleQuotedScalarString(version)
            recipe["source"][0]["sha256"] = sdist_sha256(metadata)
            recipe["build"]["number"] = 0
        with open(recipe_path, "w") as f:
            yaml.dump(recipe, f)
        logger.info(f"Wrote {recipe_path}")
    elif needs_update:
        logger.info("Recipe needs update. Re-run with --apply to write the changes.")
    else:
        logger.info("Recipe is up to date.")


if __name__ == "__main__":
    main()
