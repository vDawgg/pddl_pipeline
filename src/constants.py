from pathlib import Path


project_root = (Path(__file__) / ".." / "..").resolve()
src_dir = project_root / "src"
prompts_dir = src_dir / ".." / "prompts"
plans_dir = src_dir / ".." / "plans"
