#!/usr/bin/env bash
# ==============================================================
# AI Assistant Platform - Startup (macOS)
#
# First time only, make this executable:
#   chmod +x "Start App (Mac).command"
# Then just double-click it in Finder to launch the app.
# ==============================================================
set -u
cd "$(dirname "$0")"

echo "====================================================================="
echo "  AI Assistant Platform - Startup (macOS) (Was made by Oleh Datsyk)"
echo "====================================================================="
echo

fail() {
    echo
    echo "ERROR: $1"
    echo "The terminal will stay open so you can read this message."
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
}

# --- 1. Verify Python installation -----------------------------------------
echo "[1/7] Checking for Python 3..."
if ! command -v python3 >/dev/null 2>&1; then
    fail "Python 3 was not found. Install it from https://www.python.org/downloads/ or via 'brew install python3'."
fi
python3 --version
echo "Python found."
echo

# --- 2. Create virtual environment if needed --------------------------------
echo "[2/7] Checking virtual environment..."
if [ ! -f ".venv/bin/python3" ]; then
    echo "Creating virtual environment in .venv ..."
    python3 -m venv .venv || fail "Failed to create virtual environment."
else
    echo "Virtual environment already exists."
fi
echo

# --- 3. Activate the environment --------------------------------------------
echo "[3/7] Activating virtual environment..."
# shellcheck disable=SC1091
source ".venv/bin/activate" || fail "Failed to activate virtual environment."
echo

# --- 4. Install missing dependencies -----------------------------------------
echo "[4/7] Installing/verifying dependencies (this may take a minute on first run)..."
python3 -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt || fail "Failed to install dependencies. See the output above."
echo "Dependencies OK."
echo

# --- 5. Verify the .env file --------------------------------------------------
echo "[5/7] Checking .env file..."
if [ ! -f ".env" ]; then
    echo "No .env file found - creating one from .env.example"
    cp ".env.example" ".env"
    echo
    echo "IMPORTANT: Open the new .env file and add at least one AI provider API key"
    echo "           (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY) before chatting."
    echo
else
    echo ".env file found."
fi
echo

# --- 6. Verify required API keys ----------------------------------------------
echo "[6/7] Checking for at least one configured AI provider key..."
if python3 -c "from config import settings; import sys; sys.exit(0 if any(settings.providers_available.values()) else 1)" 2>/dev/null; then
    echo "At least one AI provider is configured."
else
    echo "WARNING: No AI provider API key is set in .env."
    echo "The app will still start, but chat features will be disabled until you add one."
fi
echo

# --- 7. Launch the application ------------------------------------------------
echo "[7/7] Launching AI Assistant Platform..."
echo "Once started, open your browser to: http://localhost:8000"
echo "Press CTRL+C in this window to stop the server."
echo
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

status=$?
if [ $status -ne 0 ]; then
    fail "The application exited with an error (exit code $status). See the output above."
fi
