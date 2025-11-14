#!/bin/sh
# Simple entrypoint for Railway
# Railway sets PORT automatically, we just need to use it

PORT=${PORT:-8080}
exec gunicorn app:app --bind "0.0.0.0:${PORT}" --workers 2 --threads 2 --timeout 120

