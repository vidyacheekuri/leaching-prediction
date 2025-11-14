#!/bin/bash
# Startup script for Railway deployment
# This ensures PORT is properly set

# Set default PORT if not provided
export PORT=${PORT:-8080}

# Start gunicorn with the PORT variable
exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app

