#!/usr/bin/env bash
# Start the TailorSync backend accessible from all network interfaces.
# Usage: ./start.sh
set -e
cd "$(dirname "$0")"
echo "Backend LAN address: http://$(ipconfig getifaddr en0):8000"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
