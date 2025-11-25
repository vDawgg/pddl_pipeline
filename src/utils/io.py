import tempfile


def write_temp_pddl_file(pddl: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    with open(tmp.name, "w") as f:
        f.write(pddl)
    return tmp.name
