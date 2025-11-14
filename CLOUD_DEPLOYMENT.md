# Quick Cloud Deployment Guide

This guide covers the **easiest** cloud deployment options for your Cement Leaching Prediction app.

## 🚀 Recommended: Render.com (Free Tier Available)

**Best for**: Quick deployment, free tier, automatic HTTPS, easy setup

### Step-by-Step:

1. **Push your code to GitHub** (if not already):
   ```bash
   # Initialize git if needed
   git init
   git add .
   git commit -m "Initial commit with trained model"
   
   # Create a new repository on GitHub, then:
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Sign up/Login to Render**:
   - Go to https://render.com
   - Sign up with GitHub (easiest)

3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

4. **Configure Settings**:
   - **Name**: `cement-leaching-prediction` (or any name)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to you
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or `.` if needed)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app`
   - **Plan**: Free (or Starter for better performance)

5. **Environment Variables** (optional):
   - `FLASK_DEBUG=false`
   - `FLASK_APP=app.py`

6. **Deploy**:
   - Click "Create Web Service"
   - Render will automatically build and deploy
   - Your app will be live at: `https://your-app-name.onrender.com`

**⚠️ Important**: Make sure your `models/` directory is committed to git:
```bash
git add models/
git commit -m "Add trained model files"
git push
```

---

## 🚂 Alternative: Railway.app (Very Simple)

**Best for**: Simplicity, automatic deployments, great developer experience

### Steps:

1. **Push to GitHub** (same as above)

2. **Deploy to Railway**:
   - Go to https://railway.app
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway auto-detects Python and uses your `Procfile`

3. **That's it!** Railway will:
   - Automatically detect Python
   - Use your `Procfile` for the start command
   - Set up HTTPS automatically
   - Deploy on every push

Your app will be live at: `https://your-app-name.up.railway.app`

---

## 🪶 Alternative: Fly.io (Global Edge Deployment)

**Best for**: Global distribution, Docker-based, free tier

### Steps:

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

3. **Initialize** (in your project directory):
   ```bash
   fly launch
   ```
   - Follow prompts
   - Choose app name
   - Select region
   - Don't deploy yet (we'll do that next)

4. **Deploy**:
   ```bash
   fly deploy
   ```

Your app will be live at: `https://your-app-name.fly.dev`

---

## ☁️ Alternative: Google Cloud Run (Serverless)

**Best for**: Pay-per-use, auto-scaling, Docker-based

### Steps:

1. **Install Google Cloud SDK**:
   ```bash
   # macOS
   brew install --cask google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Login and Set Project**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Build and Deploy**:
   ```bash
   # Build container
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cement-leaching-app
   
   # Deploy
   gcloud run deploy cement-leaching-app \
     --image gcr.io/YOUR_PROJECT_ID/cement-leaching-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8080
   ```

Your app will be live at: `https://cement-leaching-app-XXXXX-uc.a.run.app`

---

## 📋 Pre-Deployment Checklist

Before deploying, make sure:

- [ ] Model files are committed to git (`models/` directory)
- [ ] `requirements.txt` is up to date
- [ ] `Procfile` exists (for Railway)
- [ ] `.gitignore` doesn't exclude model files
- [ ] App works locally with `gunicorn`
- [ ] Environment variables are set (if needed)

## 🔍 Verify Model Files Are Included

```bash
# Check what's in git
git ls-files models/

# If models are missing, add them:
git add models/*.pkl
git commit -m "Add model files for deployment"
git push
```

## 🧪 Test After Deployment

Once deployed, test your endpoints:

```bash
# Status check
curl https://your-app-url.com/api/status

# Make a prediction
curl -X POST https://your-app-url.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "material": "Al",
    "ph": 12.0,
    "time_days": 1.0,
    "cement_type": "CEM_I",
    "form_type": "Concrete",
    "stat_measure": "CL_Minus"
  }'
```

## 🆘 Troubleshooting

### Model Not Loading
- Check logs in your platform's dashboard
- Verify model files are in the repository
- Check file paths are correct

### Build Fails
- Check `requirements.txt` is correct
- Verify Python version compatibility
- Check build logs for specific errors

### App Crashes
- Check memory limits (may need to upgrade plan)
- Verify all dependencies are in `requirements.txt`
- Check application logs

## 💰 Cost Comparison

| Platform | Free Tier | Paid Plans Start At |
|----------|-----------|---------------------|
| Render   | ✅ Yes (with limitations) | $7/month |
| Railway  | ✅ Yes (limited) | $5/month |
| Fly.io   | ✅ Yes (3 VMs) | Pay-as-you-go |
| Cloud Run| ✅ Yes (2M requests/month) | Pay-per-use |

## 🎯 Recommendation

**For beginners**: Start with **Render.com** - it's the easiest and has a good free tier.

**For Docker users**: Use **Fly.io** or **Google Cloud Run** - they work great with your existing Dockerfile.

**For simplicity**: Use **Railway** - minimal configuration needed.

---

## 📚 Additional Resources

- Full deployment guide: See `DEPLOYMENT.md`
- Docker deployment: See `Dockerfile` and `docker-compose.yml`
- API documentation: See `README.md`

