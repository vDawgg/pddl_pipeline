from pathlib import Path

from src.base.schema import PDDLFiles
from src.constants import generated_pddl_dir
from src.utils.timestamp import get_current_timestamp


def write_pddl_file(
    pddl: str,
    file: Path | None = None,
    name: str | None = None,
    pddl_file_type: PDDLFiles | None = None,
) -> Path:
    # TODO: This should probably be an override
    assert (name is not None and pddl_file_type is not None) or file is not None
    file_path = (
        file
        or generated_pddl_dir
        / f"{name}_{pddl_file_type}_{get_current_timestamp()}.pddl"
    )
    with open(file_path, "w") as f:
        f.write(pddl)
    return file_path


def read_pddl_file(pddl_file: Path) -> str:
    with open(pddl_file) as f:
        return f.read()
