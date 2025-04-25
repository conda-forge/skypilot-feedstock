#! /usr/bin/env python3

# This is an auxiliary script to check if the recipe is up to date.
# It is not used in the recipe itself.

from typing import List, Sequence
from ruamel.yaml import YAML
from pathlib import Path
import requests
import logging
from packaging.requirements import Requirement
import re
from conda_lock.lookup import pypi_name_to_conda_name, DEFAULT_MAPPING_URL
from conda_lock.src_parser.pyproject_toml import poetry_version_to_conda_version

current_dir = Path(__file__).parent
recipe_path = current_dir / "recipe.yaml"

yaml = YAML(typ="safe")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("update")


def is_required_for_extras(requirement: Requirement, extras: Sequence[str]) -> bool:
    if not requirement.marker:
        # No marker means it's a core requirement
        return True
    if requirement.marker.evaluate({}):
        # There's a marker, but it's satisfied by the empty extras
        return True
    for extra in extras:
        env = {"extra": extra}
        if requirement.marker.evaluate(env):
            return True
    return False


def patch_upstream_requirements(requirements: Sequence[str]) -> Sequence[str]:
    """
    These markers are complicated to evaluate:
        grpcio!=1.48.0,<=1.49.1,>=1.32.0; (python_version < "3.10" and sys_platform == "darwin") and extra == "all"
        grpcio!=1.48.0,<=1.49.1,>=1.32.0; (python_version < "3.10" and sys_platform == "darwin") and extra == "remote"
        grpcio!=1.48.0,<=1.49.1,>=1.42.0; (python_version >= "3.10" and sys_platform == "darwin") and extra == "remote"
        grpcio!=1.48.0,<=1.49.1,>=1.42.0; (python_version >= "3.10" and sys_platform == "darwin") and extra == "all"
        grpcio!=1.48.0,<=1.51.3,>=1.42.0; (python_version >= "3.10" and sys_platform != "darwin") and extra == "remote"
        grpcio!=1.48.0,<=1.51.3,>=1.42.0; (python_version >= "3.10" and sys_platform != "darwin") and extra == "all"
    We remove the part that looks like this:
        (python_version >= "3.10" and sys_platform != "darwin") and
    Then we are left with combination of all the above requirements which is still satisfiable.

    On conda-forge we have ray[default] provided by ray-default.
    """
    result = [
        re.sub(r'\(python_version .+ "3\.10" and sys_platform .= "darwin"\) and ', '', req)
        for req in requirements
    ]
    result = [
        re.sub(r'^ray\[default\]', 'ray-default', req)
        for req in result
    ]
    return result


if __name__ == "__main__":
    with open(recipe_path, "r") as f:
        recipe = yaml.load(f)

    name = recipe["recipe"]["name"]
    version = recipe["context"]["version"]
    logger.info(f"Current {name} version: {version}")

    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    logger.info(f"Fetching metadata from {metadata_url}")
    response = requests.get(metadata_url)
    metadata = response.json()

    if not metadata["info"]["name"] == name:
        raise ValueError(
            f"Current name {name} does not match metadata name {metadata['info']['name']}"
        )
    if not metadata["info"]["version"] == version:
        raise ValueError(
            f"Current version {version} does not match metadata version {metadata['info']['version']}"
        )
    extras = metadata["info"]["provides_extra"]

    outputs = {output["package"]["name"]: output for output in recipe["outputs"]}
    assert name in outputs, f"{name} not found in outputs"
    outputs_for_extras = {}
    for output in outputs.keys():
        if output == name:
            outputs_for_extras[None] = outputs[output]
            continue
        if not output.startswith(f"{name}-"):
            raise ValueError(f"{output} is not an expected output for {name}.")
        output_suffix = output[len(name) + 1 :]
        if output_suffix not in extras:
            raise ValueError(f"{output_suffix} is not an expected extra for {name}.")
        outputs_for_extras[output_suffix] = outputs[output]
    extras_without_outputs = set(extras) - set(outputs_for_extras.keys()) - {None}
    logger.info(f"Extras without outputs: {extras_without_outputs}")

    raw_upstream_requirements = metadata["info"]["requires_dist"]
    patched_upstream_requirements = patch_upstream_requirements(raw_upstream_requirements)
    upstream_requirements = [
        Requirement(req) for req in patched_upstream_requirements
    ]
    core_requirements = [
        req for req in upstream_requirements if is_required_for_extras(req, [])
    ]
    logger.debug(f"Core requirements: {[str(req) for req in core_requirements]}")
    requirements_for_extras = {
        extra: [
            req
            for req in upstream_requirements
            if req not in core_requirements and is_required_for_extras(req, [extra])
        ]
        for extra in extras
    }
    logger.debug(f"Requirements for extras: {requirements_for_extras}")
    core_and_extras = set(core_requirements) | set(
        sum(requirements_for_extras.values(), [])
    )
    if not set(upstream_requirements) == core_and_extras:
        logger.error(
            "Upstream requirements do not match core requirements and requirements for extras:"
        )
        unexpected_requirements = core_and_extras - set(upstream_requirements)
        missing_requirements = set(upstream_requirements) - core_and_extras
        for req in unexpected_requirements:
            logger.error(f"Unexpected requirement: {req}")
        for req in missing_requirements:
            logger.error(f"Missing requirement: {req}")
        raise ValueError("Inconsistent requirements.")

    for extra in extras_without_outputs:
        if extra == "all":
            logger.info("Skipping 'all' extra because it's excessive.")
            continue
        if extra in ["cudo", "ibm", "azure", "runpod", "do", "vast", "nebius"]:
            logger.info(f"Skipping '{extra}' extra because it's not on conda-forge.")
            continue
        if len(requirements_for_extras[extra]) == 0:
            logger.info(f"Skipping {extra} extra because it has no requirements.")
            continue
        raise ValueError(
            f"Missing extra {extra} in outputs. Requirements: {requirements_for_extras[extra]}"
        )

    needs_update = False
    for extra, output_recipe in outputs_for_extras.items():
        if extra is None:
            expected_requirements: List[Requirement] = core_requirements
            print("Checking core requirements")
        else:
            expected_requirements = requirements_for_extras[extra]
            print(f"Checking '{extra}' extra requirements")
        current_requirements = output_recipe["requirements"]["run"]
        current_requirements = [r for r in current_requirements if not r.startswith("python ")]
        if extra is not None:
            if current_requirements[-1] != "${{ pin_subpackage('" + name + "', exact=True) }}":
                raise ValueError(f"Last requirement '{current_requirements[-1]}' is not a pin of the package.")
            current_requirements = current_requirements[:-1]
        for expected, current in zip(expected_requirements, current_requirements):
            if expected.extras:
                raise NotImplementedError(f"Contains extras: {expected}")
            if expected.marker:
                if not re.match(r"extra == ['\"].*['\"]", str(expected.marker)):
                    raise NotImplementedError(f"Marker is not supported: {expected}")
            conda_name = pypi_name_to_conda_name(expected.name, mapping_url=DEFAULT_MAPPING_URL)
            conda_version = poetry_version_to_conda_version(str(expected.specifier))
            expected_value = f"{conda_name} {conda_version}".rstrip()
            if expected_value != current:
                print(f"Update '{current}' to '{expected_value}'")
                needs_update = True
        if len(expected_requirements) != len(current_requirements):
            needs_update = True
            if len(expected_requirements) > len(current_requirements):
                print(f"Add {expected_requirements[len(current_requirements):]}")
            else:
                print(f"Remove {current_requirements[len(expected_requirements):]}")
    if needs_update:
        logger.info("Recipe needs update.")
    else:
        logger.info("Recipe is up to date.")
