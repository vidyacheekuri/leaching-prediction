web: python -c "import os, subprocess; subprocess.run(['gunicorn', 'app:app', '--bind', f\"0.0.0.0:{os.environ.get('PORT', '8080')}\", '--workers', '2', '--threads', '2', '--timeout', '120'])"

