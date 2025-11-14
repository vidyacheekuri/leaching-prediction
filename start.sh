#!/bin/bash
# Startup script for Railway deployment
# This ensures PORT is properly set

# Set default PORT if not provided (Railway sets this automatically)
PORT=${PORT:-8080}

# Validate PORT is a number
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: PORT must be a number, got: $PORT"
    exit 1
fi

# Start gunicorn with the PORT variable
exec gunicorn --bind "0.0.0.0:${PORT}" --workers 2 --threads 2 --timeout 120 app:app

