#!/usr/bin/env python
"""
run_dashboard.py — one command to run the UK inflation nowcasting workstation (Part J).

    python run_dashboard.py                 # refresh data, then launch the terminal
    python run_dashboard.py --no-refresh     # launch on existing snapshot (instant)
    python run_dashboard.py --refresh-only    # update data only (for cron/launchd)
    python run_dashboard.py --only headline   # refresh a subset

Steps: (1) update factors + HF + rerun intramonth pipeline via dashboard.refresh,
(2) launch the Streamlit app. FRED_API_KEY is read from the environment or ./.env.
"""
import os, sys, argparse, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")
APP = os.path.join(CODE, "dashboard", "app.py")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
PY = VENV_PY if os.path.exists(VENV_PY) else sys.executable


def _load_env():
    """Load ./.env into os.environ if present (FRED_API_KEY etc.)."""
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def refresh(end_year, only):
    env = dict(os.environ, PYTHONPATH=CODE)
    cmd = [PY, "-m", "dashboard.refresh", "--end-year", str(end_year)]
    if only:
        cmd += ["--only", only]
    print(">> refreshing nowcast data …")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def launch():
    env = dict(os.environ, PYTHONPATH=CODE)
    print(">> launching Streamlit terminal (Ctrl-C to stop) …")
    subprocess.run([PY, "-m", "streamlit", "run", APP,
                    "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                   cwd=ROOT, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="skip data refresh, launch only")
    ap.add_argument("--refresh-only", action="store_true", help="update data only, no UI")
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--only", default=None, help="subset: headline,core,services")
    args = ap.parse_args()

    _load_env()
    if not args.no_refresh:
        refresh(args.end_year, args.only)
    if not args.refresh_only:
        launch()


if __name__ == "__main__":
    main()
