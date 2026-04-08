import os
import re
import sys
from pathlib import Path
from random import shuffle

import dspy

from src.base.schemas import Prompts
from src.constants import (
    pddlgym_action_prompts_dir,
    pddlgym_domain_prompts_dir,
    pddlgym_object_names_prompts_dir,
    pddlgym_problem_prompts_dir,
)
from src.utils.prompts import get_prompt

# Suppress gym's deprecation notice as we do not require newer functionality and have no
# reason to upgrade gyms version
_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
try:
    import pddlgym
    from pddlgym.core import PDDLEnv
    from pddlgym.parser import parse_plan_step
finally:
    sys.stderr.close()
    sys.stderr = _stderr


def get_pddl_gym_domain_prompt(file_name: str) -> str:
    with open(pddlgym_domain_prompts_dir / file_name) as f:
        return f.read()


def get_pddl_gym_problem_prompt(file_name: str) -> str:
    with open(pddlgym_problem_prompts_dir / file_name) as f:
        return f.read()


def get_action_schema_prompt(file_name: str) -> str:
    with open(pddlgym_action_prompts_dir / file_name) as f:
        return f.read()


def get_object_names_prompt(file_name: str) -> str:
    with open(pddlgym_object_names_prompts_dir / file_name) as f:
        return f.read()


def get_combined_pddlgym_prompt(domain: str, problem: str) -> str:
    with open(pddlgym_domain_prompts_dir / domain) as f:
        domain = f.read()
    with open(pddlgym_problem_prompts_dir / problem) as f:
        problem = f.read()
    return get_prompt(Prompts.DOMAIN_AND_PROBLEM).format(
        domain=domain,
        problem=problem,
    )


def make_example(
    domain_prompt_file: str,
    problem_prompt_file: str,
    action_schema_prompt_file: str,
    object_names_prompt_file: str,
    domain_name: str,
    problem_idx: int,
    separate_prompts: bool,
):
    if separate_prompts:
        return dspy.Example(
            domain_description=get_pddl_gym_domain_prompt(
                domain_prompt_file,
            ),
            problem_description=get_pddl_gym_problem_prompt(
                problem_prompt_file,
            ),
            domain_name=domain_name,
            problem_index=problem_idx,
        ).with_inputs(
            "domain_description",
            "problem_description",
        )
    return dspy.Example(
        task_description=get_combined_pddlgym_prompt(
            domain_prompt_file,
            problem_prompt_file,
        ),
        action_schema=get_action_schema_prompt(action_schema_prompt_file),
        object_names=get_object_names_prompt(object_names_prompt_file),
        domain_name=domain_name,
        problem_index=problem_idx,
    ).with_inputs("task_description", "action_schema")


def make_ds(
    split: tuple[int, int] = (90, 10),
    separate_prompts: bool = False,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Make dataset for prompt optimization from natural language descriptions of
    pddlgym tasks.
    Can be formatted for tool-call or rigid-trajectory pipelines. Set separate
    prompts to True to format for rigid-trajectory.

    returns randomly sampled and shuffled train and val splits
    """
    # blocks
    blocks = [
        make_example(
            domain_prompt_file="Blocks.md",
            problem_prompt_file=f"Blocks_{i}.md",
            action_schema_prompt_file="Blocks.md",
            object_names_prompt_file="Blocks.md",
            domain_name="PDDLEnvBlocks",
            problem_idx=i,
            separate_prompts=separate_prompts,
        )
        for i in range(5)
    ]
    blocks.extend(
        [
            make_example(
                domain_prompt_file="BlocksMedium.md",
                problem_prompt_file=f"BlocksMedium_{i}.md",
                action_schema_prompt_file="BlocksMedium.md",
                object_names_prompt_file="BlocksMedium.md",
                domain_name="PDDLEnvBlocks_medium",
                problem_idx=i,
                separate_prompts=separate_prompts,
            )
            for i in range(40)
        ]
    )
    # gripper
    gripper = [
        make_example(
            domain_prompt_file="Gripper.md",
            problem_prompt_file=f"Gripper_{i}.md",
            action_schema_prompt_file="Gripper.md",
            object_names_prompt_file="Gripper.md",
            domain_name="PDDLEnvGripper",
            problem_idx=i,
            separate_prompts=separate_prompts,
        )
        for i in range(10)
    ]
    gripper.extend(
        [
            make_example(
                domain_prompt_file="Gripper.md",
                problem_prompt_file=f"ManyGripper_{i}.md",
                action_schema_prompt_file="Gripper.md",
                object_names_prompt_file="Gripper.md",
                domain_name="PDDLEnvManygripper",
                problem_idx=i,
                separate_prompts=separate_prompts,
            )
            for i in range(40)
        ]
    )
    # search and rescue
    search_and_rescue = [
        make_example(
            domain_prompt_file="SearchAndRescue.md",
            problem_prompt_file=f"SearchAndRescueLevel1_{i}.md",
            action_schema_prompt_file="SearchAndRescue.md",
            object_names_prompt_file="SearchAndRescue.md",
            domain_name="PDDLSearchAndRescueLevel1",
            problem_idx=i,
            separate_prompts=separate_prompts,
        )
        for i in range(20)
    ]
    search_and_rescue.extend(
        [
            make_example(
                domain_prompt_file="SearchAndRescue.md",
                problem_prompt_file=f"SearchAndRescueLevel2_{i}.md",
                action_schema_prompt_file="SearchAndRescue.md",
                object_names_prompt_file="SearchAndRescue.md",
                domain_name="PDDLSearchAndRescueLevel2",
                problem_idx=i,
                separate_prompts=separate_prompts,
            )
            for i in range(50)
        ]
    )
    # depot
    depot = [
        make_example(
            domain_prompt_file="Depot.md",
            problem_prompt_file=f"Depot_{i}.md",
            action_schema_prompt_file="Depot.md",
            object_names_prompt_file="Depot.md",
            domain_name="PDDLEnvDepot",
            problem_idx=i,
            separate_prompts=separate_prompts,
        )
        for i in range(10)
    ]
    depot.extend(
        make_example(
            domain_prompt_file="Depot.md",
            problem_prompt_file=f"DepotTest_{i}.md",
            action_schema_prompt_file="Depot.md",
            object_names_prompt_file="Depot.md",
            domain_name="PDDLEnvDepotTest",
            problem_idx=i,
            separate_prompts=separate_prompts,
        )
        for i in range(12)
    )
    # minecraft
    minecraft = [
        make_example(
            domain_prompt_file="Minecraft.md",
            problem_prompt_file=f"Minecraft_{i}.md",
            action_schema_prompt_file="Minecraft.md",
            object_names_prompt_file="Minecraft.md",
            domain_name="PDDLEnvMinecraft",
            problem_idx=i,
            separate_prompts=separate_prompts,
        )
        for i in range(30)
    ]
    minecraft.extend(
        make_example(
            domain_prompt_file="Minecraft.md",
            problem_prompt_file=f"MinecraftTest_{i}.md",
            action_schema_prompt_file="Minecraft.md",
            object_names_prompt_file="Minecraft.md",
            domain_name="PDDLEnvMinecraftTest",
            problem_idx=i,
            separate_prompts=separate_prompts,
        )
        for i in range(30)
    )

    shuffle(blocks)
    shuffle(gripper)
    shuffle(search_and_rescue)
    shuffle(depot)
    shuffle(minecraft)

    val_split = split[1] / 100

    blocks_val_split = int(len(blocks) * val_split)
    gripper_val_split = int(len(gripper) * val_split)
    search_and_rescue_val_split = int(len(search_and_rescue) * val_split)
    depot_val_split = int(len(depot) * val_split)
    minecraft_val_split = int(len(minecraft) * val_split)

    val = (
        blocks[:blocks_val_split]
        + gripper[:gripper_val_split]
        + search_and_rescue[:search_and_rescue_val_split]
        + depot[:depot_val_split]
        + minecraft[:minecraft_val_split]
    )
    train = (
        blocks[blocks_val_split:]
        + gripper[gripper_val_split:]
        + search_and_rescue[search_and_rescue_val_split:]
        + depot[depot_val_split:]
        + minecraft[minecraft_val_split:]
    )

    shuffle(val)
    shuffle(train)

    return train, val


def goal_reached(domain_name: str, problem_index: int, plan_file: Path) -> bool:
    env: PDDLEnv = pddlgym.make(f"{domain_name}-v0")  # type: ignore
    env.fix_problem_index(problem_index)
    state, _ = env.reset()
    with open(plan_file) as f:
        lines = f.readlines()
    plan = []
    for line in lines:
        match = re.match(
            r"^\((?P<function>\S+)\s+(?P<parameters>.+)\)$",
            line,
        )
        if match:
            action = match.group("function")
            # Action name is not known in current domain
            if action not in env.domain.operators:
                return False
            parameters = match.group("parameters").split(" ")
            # Actions parameter count does not fulfill lower bound.
            if len(parameters) < len(env.domain.operators[action].params):
                return False
            # This is just a best effort assumption, as we cannot easilly validate whether
            # the needed parameters are given in the correct order
            parameters = parameters[: len(env.domain.operators[action].params)]
            plan.append(f"{action} {' '.join(parameters)}")
    try:
        plan = [
            parse_plan_step(
                plan_step,
                env.domain.operators.values(),
                [env.domain.predicates[a] for a in list(env.domain.actions)],
                state.objects,  # type: ignore
                operators_as_actions=env.domain.operators_as_actions,
            )
            for plan_step in plan
        ]
    except AssertionError:
        return False
    terminated = False
    while not terminated:
        try:
            action = plan.pop(0)
        except IndexError:
            return False
        state, _, terminated, _, _ = env.step(action)
    return True
