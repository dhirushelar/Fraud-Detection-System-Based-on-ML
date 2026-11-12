#!/bin/sh
# ==============================================================================
# Fraud Detection System (FDS) - Container Startup Script
# Author: Alpha
# ==============================================================================

set -e

echo "[+] Starting Fraud Detection System (FDS)..."

# Ensure artifacts exist, train fallback if first boot
if [ ! -f "ml/artifacts/isolation_forest_v1.0.0.joblib" ]; then
    echo "[!] Model artifacts not detected in volume. Initializing baseline training..."
    python scripts/train_model.py || true
fi

# Seed database with initial RBAC and mock transactions if DB is empty
echo "[+] Checking database seeding..."
python scripts/seed_db.py || true

echo "[✓] Environment initialized. Launching Gunicorn WSGI server..."
exec gunicorn --bind 0.0.0.0:5000 \
    --workers 4 \
    --threads 2 \
    --timeout 60 \
    --access-logfile - \
    --error-logfile - \
    wsgi:app
