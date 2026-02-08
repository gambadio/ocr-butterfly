#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of OCR Butterfly.
# Based on MLX-Video-OCR-DeepSeek-Apple-Silicon by matica0902 (AGPL-3.0).
# Copyright (C) 2025 MLX DeepSeek-OCR contributors
# Copyright (C) 2026 Ricardo Kupper
# See the LICENSE file in the project root for full license text.

# OCR Butterfly start script (auto port selection + zombie cleanup)

echo "Starting OCR Butterfly..."
echo ""

cd "$(dirname "$0")"

echo "Checking Python version..."
python3 --version

# Virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment found, activating..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
MISSING=false
python3 -c "import flask" 2>/dev/null || MISSING=true
python3 -c "import mlx_vlm" 2>/dev/null || MISSING=true
python3 -c "from PIL import Image" 2>/dev/null || MISSING=true
python3 -c "import fitz" 2>/dev/null || MISSING=true
python3 -c "import cv2" 2>/dev/null || MISSING=true
python3 -c "import transformers" 2>/dev/null || MISSING=true

if [ "$MISSING" = true ]; then
    echo "Missing dependencies, installing..."
    pip install -r requirements.txt
    [ $? -eq 0 ] && echo "Dependencies installed successfully" || { echo "Installation failed"; exit 1; }
else
    echo "All dependencies installed"
fi

# Find available port + kill zombie processes
PORT=5001
MAX_PORT=5010
echo ""
echo "Finding available port and cleaning up zombie processes..."

while [ $PORT -le $MAX_PORT ]; do
    if lsof -i:$PORT >/dev/null 2>&1; then
        echo "Port $PORT is in use, attempting to kill process..."
        lsof -i:$PORT | grep python | awk '{print $2}' | xargs kill -9 2>/dev/null || true
        if lsof -i:$PORT >/dev/null 2>&1; then
            echo "Cannot free port $PORT, trying next..."
            ((PORT++))
        else
            echo "Successfully freed and using port: $PORT"
            break
        fi
    else
        echo "Found available port: $PORT"
        break
    fi
done

if [ $PORT -gt $MAX_PORT ]; then
    echo "Error: No available port found (5001-5010)"
    exit 1
fi

echo ""
echo "Starting OCR Butterfly (port passed via env, overrides app.py default)"
echo "Access URL: http://localhost:$PORT"
echo "Press Ctrl+C to stop safely"
echo ""

# Listen on all interfaces + disable reload (prevents duplicate processes)
PORT=$PORT python3 app.py
