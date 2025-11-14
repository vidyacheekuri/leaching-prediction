#!/bin/bash
# Quick script to prepare and deploy to cloud

set -e

echo "☁️  Cloud Deployment Preparation"
echo "================================"
echo ""

# Check if model files exist
if [ ! -f "models/production_model.pkl" ]; then
    echo "❌ Error: Model files not found!"
    echo "   Please train the model first: python main.py"
    exit 1
fi

echo "📦 Step 1: Adding model files to git..."
# Force add model files (even if in .gitignore)
git add -f models/production_model*.pkl

echo "✅ Model files added"
echo ""

echo "📝 Step 2: Committing changes..."
git add app.py requirements.txt Dockerfile Procfile docker-compose.yml render.yaml .dockerignore DEPLOYMENT.md CLOUD_DEPLOYMENT.md
git commit -m "Add deployment configuration and model files" || echo "No new changes to commit"

echo ""
echo "🚀 Step 3: Ready to deploy!"
echo ""
echo "Choose your platform:"
echo ""
echo "1. Render.com (Recommended - Free tier):"
echo "   → Go to https://render.com"
echo "   → New Web Service → Connect GitHub repo"
echo "   → Build: pip install -r requirements.txt"
echo "   → Start: gunicorn --bind 0.0.0.0:\$PORT --workers 2 --threads 2 --timeout 120 app:app"
echo ""
echo "2. Railway.app (Easiest):"
echo "   → Go to https://railway.app"
echo "   → New Project → Deploy from GitHub"
echo "   → Railway auto-detects everything!"
echo ""
echo "3. Fly.io (Docker-based):"
echo "   → fly launch (in this directory)"
echo ""
echo "📚 See CLOUD_DEPLOYMENT.md for detailed instructions"
echo ""
read -p "Push to GitHub now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Pushing to GitHub..."
    git push origin main || git push origin master
    echo "✅ Pushed! Now go to your chosen platform and deploy."
else
    echo "⏭️  Skipping push. Run 'git push' when ready."
fi

echo ""
echo "✨ Done! Your app is ready for cloud deployment."

