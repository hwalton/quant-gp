# import subprocess
# import sys
# import os

# def run_script(script_path, working_dir):
#     print(f"\n[Running] {script_path}")
#     result = subprocess.run(
#         [sys.executable, script_path],
#         cwd=working_dir,
#         capture_output=True,
#         text=True
#     )
#     print(result.stdout)
#     if result.returncode != 0:
#         print(result.stderr)
#         raise RuntimeError(f"Script {script_path} failed with exit code {result.returncode}")

# def main():
#     project_root = os.path.dirname(os.path.abspath(__file__))

#     steps = [
#         ('b_log_fit/log_fit.py', 'b_log_fit'),
#         # ('c_gp_fit/gp_fit.py', 'c_gp_fit'),
#         ('c_gp_fit/online_gp_fit.py', 'c_gp_fit'),
#         ('d_optimise_portfolio/optimise_portfolio.py', 'd_optimise_portfolio'),
#     ]

#     for script_rel, work_dir_rel in steps:
#         script_path = os.path.join(project_root, script_rel)
#         working_dir = os.path.join(project_root, work_dir_rel)
#         run_script(script_path, working_dir)

# if __name__ == "__main__":
#     main()

from a_data.format_data import main as format_data_main
from b_log_fit.log_fit import main as log_fit_main
from c_gp_fit.online_gp_fit import main as online_gp_fit_main

def main():
    format_data_main()
    log_fit_main()
    online_gp_fit_main()

if __name__ == "__main__":
    main()
