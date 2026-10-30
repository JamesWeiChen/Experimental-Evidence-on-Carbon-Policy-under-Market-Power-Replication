#!/usr/bin/env python3
"""Master script: rebuilds every table and figure from the de-identified raw data.

Usage (from this folder):
    python run_all.py

Order:
  1. build/code     raw oTree exports (build/input)  -> panels (build/output, analysis/input)
  2. analysis/code  panels (analysis/input)          -> tables/figures (analysis/output)

Each script's console output is saved to logs/<script>.log.
"""
import glob
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")

BUILD_SCRIPTS = [
    "gen_subj_and_group_panel.py",
    "gen_trade_panel.py",
]

ANALYSIS_SCRIPTS = [
    "gen_firm_net_trade_plot.py",
    "gen_trade_seconds_vs_price_diff.py",
    "run_group_lv_reg.py",
    "run_subj_lv_reg.py",
    "run_trade_mkt_reg.py",
    "gen_summary_table_cells.py",  # added for the package; needs run_group_lv_reg.py and run_subj_lv_reg.py outputs
]


def remove_stale_outputs(root_dir, patterns):
    for pattern in patterns:
        for path in glob.glob(os.path.join(root_dir, pattern)):
            if os.path.isfile(path):
                os.remove(path)
                print(f"Removed stale output: {path}")


def run(script_path):
    name = os.path.splitext(os.path.basename(script_path))[0]
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    print(f"Running: {os.path.relpath(script_path, ROOT)}")
    start = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, script_path], check=True, stdout=log,
                       stderr=subprocess.STDOUT, cwd=os.path.dirname(script_path), env=env)
    print(f"  done in {time.time() - start:.1f}s (log: {os.path.relpath(log_path, ROOT)})")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Python {platform.python_version()} on {platform.platform()}")
    t0 = time.time()

    for script in BUILD_SCRIPTS:
        run(os.path.join(ROOT, "build", "code", script))

    remove_stale_outputs(os.path.join(ROOT, "analysis", "output"), ["subj_lv_net_trade*"])

    for script in ANALYSIS_SCRIPTS:
        run(os.path.join(ROOT, "analysis", "code", script))

    print(f"All scripts finished in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
