import subprocess
import sys
import os

def run_script(script_path, working_dir):
    print(f"\n[Running] {script_path}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=working_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Script {script_path} failed with exit code {result.returncode}")

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    steps = [
        ('1-log-fit/log-fit.py', '1-log-fit'),
        # ('2-gp_fit/gp_fit.py', '2-gp_fit'),
        ('2-gp_fit/online_gp_fit.py', '2-gp_fit'),
        ('3-optimise_portfolio/optimise_portfolio.py', '3-optimise_portfolio'),
    ]

    for script_rel, work_dir_rel in steps:
        script_path = os.path.join(project_root, script_rel)
        working_dir = os.path.join(project_root, work_dir_rel)
        run_script(script_path, working_dir)

if __name__ == "__main__":
    main()
