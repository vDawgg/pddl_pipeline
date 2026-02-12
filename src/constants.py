from pathlib import Path

project_root = (Path(__file__) / ".." / "..").resolve()
src_dir = project_root / "src"
prompts_dir = project_root / "prompts"
plans_dir = project_root / "plans"
generated_pddl_dir = project_root / "pddl" / "generated"
results_dir = project_root / "results"
logs_dir = project_root / "logs"
images_dir = project_root / "images"
charts_dir = project_root / "charts"
