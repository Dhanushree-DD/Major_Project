#!/usr/bin/env bash

# Navigate to script directory
cd "$(dirname "$0")"

echo "======================================================================"
echo "          🧠 FetalBrain AI - Fetal Neurosonogram System"
echo "======================================================================"
echo ""

# 1. Detect Python
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 was not found on your system."
    echo "Please install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
fi

echo "[✓] Python detected: $($PY_CMD --version)"
echo ""

# 2. Setup Virtual Environment if not exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] First time setup detected! Creating virtual environment 'venv'..."
    $PY_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
    source venv/bin/activate
    echo "[INFO] Installing required dependencies (PyTorch, Timm, Flask, ReportLab)..."
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Dependency installation encountered an issue."
        exit 1
    fi
    echo "[✓] Dependencies installed successfully!"
    echo ""
else
    source venv/bin/activate
fi

# 3. Launch Flask Web Server & Auto-open Browser
echo "======================================================================"
echo "[INFO] Starting FetalBrain AI Web Server..."
echo "[INFO] Portal URL: http://127.0.0.1:5000"
echo ""
echo "[INFO] Admin Credentials:"
echo "       - Login Tab:  Admin Portal"
echo "       - Username:   admin"
echo "       - Password:   admin123"
echo "======================================================================"
echo ""

# Auto-open browser in background
if command -v open &> /dev/null; then
    (sleep 2 && open "http://127.0.0.1:5000") &
elif command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open "http://127.0.0.1:5000") &
fi

python3 webapp/app.py
