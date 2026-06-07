#!/usr/bin/env bash
# Auto-organize output files dropped in project root into subfolders.
# Runs as a Claude Code Stop hook after each session.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

find "$ROOT" -maxdepth 1 -name "*.py"                         -exec mv {} "$ROOT/code/"  \;
find "$ROOT" -maxdepth 1 -name "*.csv"                        -exec mv {} "$ROOT/data/"  \;
find "$ROOT" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) -exec mv {} "$ROOT/plots/" \;
find "$ROOT" -maxdepth 1 -name "*.log"                        -exec mv {} "$ROOT/logs/"  \;
# Move markdown docs to docs/ — README.md stays in root (GitHub convention)
find "$ROOT" -maxdepth 1 -name "*.md" ! -name "README.md"     -exec mv {} "$ROOT/docs/"  \;
