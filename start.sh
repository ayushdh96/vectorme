#!/bin/bash
# Start the VectorMe server with GPU support

cd "$(dirname "$0")"

exec vectorme --serve --gpu --host 0.0.0.0 --port 3120
