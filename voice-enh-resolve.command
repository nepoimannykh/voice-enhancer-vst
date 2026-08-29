#!/bin/sh
# DaVinci Resolve External Audio Process command-line entry point.
# Resolve appends the bounced audio path as an argument.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$PROJECT_DIR/bin/voice-enh-resolve" "$@"
