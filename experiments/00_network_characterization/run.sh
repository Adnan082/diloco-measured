#!/usr/bin/env bash
# run.sh — 00_network_characterization. Thin wrapper around the CLI (no logic here, mirrors
# cli.py's own rule — CLAUDE.md §19.1).
set -euo pipefail
diloco-measured network characterize --levels 5g,1g,200m,50m
