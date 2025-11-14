# Deployment Guide

This guide covers multiple deployment options for the Cement Leaching Prediction Flask application.

## Prerequisites

Before deploying, ensure you have:
1. ✅ Trained and saved the model by running `python main.py`
2. ✅ Verified the model files exist in the `models/` directory:
   - `production_model.pkl`
   - `production_model_metadata.pkl`
   - `production_model_transformers.pkl`
   - `production_model_label_encoders.pkl`
   - `production_model_feature_columns.pkl`

## Deployment Options

### 1. Docker Deployment (Recommended for Production)

Docker provides a consistent environment across different platforms.

#### Build and Run Locally

```bash
# Build the Docker image
docker build -t cement-leaching-app .

# Run the container
docker run -p 8080:8080 cement-leaching-app
```

#### Using Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

#### Deploy to Cloud Platforms

**Docker Hub / Container Registry:**
```bash
# Tag the image
docker tag cement-leaching-app yourusername/cement-leaching-app:latest

# Push to registry
docker push yourusername/cement-leaching-app:latest
```

Then deploy to:
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **DigitalOcean App Platform**
- **Fly.io**

---

### 2. Render.com Deployment

[Render.com](https://render.com) offers free tier hosting with automatic deployments.

#### Steps:

1. **Push your code to GitHub/GitLab/Bitbucket**

2. **Create a new Web Service on Render:**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your repository

3. **Configure the service:**
   - **Name**: `cement-leaching-prediction`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app`
   - **Plan**: Free tier (or paid for better performance)

4. **Environment Variables** (optional, set in Render dashboard):
   - `FLASK_DEBUG=false`
   - `FLASK_APP=app.py`

5. **Deploy**: Render will automatically deploy on every push to your main branch

**Note**: Make sure your `models/` directory is committed to git or use Render's persistent disk feature.

---

### 3. Railway.app Deployment

[Railway](https://railway.app) provides simple deployment with automatic HTTPS.

#### Steps:

1. **Install Railway CLI** (optional):
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Deploy via Dashboard:**
   - Go to https://railway.app
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect Python and use the `Procfile`

3. **Or deploy via CLI:**
   ```bash
   railway init
   railway up
   ```

4. **Set Environment Variables** (if needed):
   ```bash
   railway variables set FLASK_DEBUG=false
   ```

The `Procfile` will automatically be used. Railway sets the `PORT` environment variable automatically.

---

### 4. Heroku Deployment

[Heroku](https://heroku.com) is a popular PaaS platform.

#### Steps:

1. **Install Heroku CLI:**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Or download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and create app:**
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Set environment variables:**
   ```bash
   heroku config:set FLASK_DEBUG=false
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

5. **Open the app:**
   ```bash
   heroku open
   ```

**Note**: Heroku's free tier has been discontinued. Consider other options for free hosting.

---

### 5. AWS Elastic Beanstalk

AWS Elastic Beanstalk simplifies AWS deployment.

#### Steps:

1. **Install EB CLI:**
   ```bash
   pip install awsebcli
   ```

2. **Initialize EB:**
   ```bash
   eb init -p python-3.10 cement-leaching-app
   ```

3. **Create environment:**
   ```bash
   eb create cement-leaching-env
   ```

4. **Deploy:**
   ```bash
   eb deploy
   ```

5. **Open:**
   ```bash
   eb open
   ```

---

### 6. Google Cloud Run

Google Cloud Run is serverless and scales automatically.

#### Steps:

1. **Install Google Cloud SDK:**
   ```bash
   # Follow: https://cloud.google.com/sdk/docs/install
   ```

2. **Build and push container:**
   ```bash
   # Set your project
   gcloud config set project YOUR_PROJECT_ID
   
   # Build and push
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cement-leaching-app
   ```

3. **Deploy:**
   ```bash
   gcloud run deploy cement-leaching-app \
     --image gcr.io/YOUR_PROJECT_ID/cement-leaching-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8080
   ```

---

### 7. DigitalOcean App Platform

DigitalOcean's App Platform is simple and scalable.

#### Steps:

1. **Go to DigitalOcean Dashboard:**
   - Navigate to Apps → Create App

2. **Connect Repository:**
   - Connect your GitHub/GitLab repository

3. **Configure:**
   - **Type**: Web Service
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app`
   - **Environment Variables**: Set `FLASK_DEBUG=false`

4. **Deploy**: DigitalOcean will automatically deploy

---

### 8. Fly.io Deployment

[Fly.io](https://fly.io) offers global edge deployment.

#### Steps:

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login:**
   ```bash
   fly auth login
   ```

3. **Create app:**
   ```bash
   fly launch
   ```

4. **Deploy:**
   ```bash
   fly deploy
   ```

---

## Environment Variables

You can configure the app using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Port to run the server on |
| `HOST` | `0.0.0.0` | Host to bind to |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |
| `FLASK_APP` | `app.py` | Flask application file |

## Production Checklist

Before deploying to production:

- [ ] Set `FLASK_DEBUG=false` in production
- [ ] Ensure model files are included in deployment
- [ ] Test the `/api/status` endpoint
- [ ] Configure proper logging
- [ ] Set up monitoring/alerting
- [ ] Configure HTTPS/SSL (most platforms do this automatically)
- [ ] Set up backup strategy for model files
- [ ] Configure CORS if needed for API access
- [ ] Set resource limits (memory, CPU)
- [ ] Test the deployment with sample predictions

## Health Check

The app includes a health check endpoint at `/api/status`. Most platforms will automatically use this for health monitoring.

Test it locally:
```bash
curl http://localhost:8080/api/status
```

## Troubleshooting

### Model Not Loading

If you see "Model not loaded" errors:

1. **Check model files exist:**
   ```bash
   ls -la models/production_model*.pkl
   ```

2. **Verify file paths in deployment:**
   - Ensure `models/` directory is included in deployment
   - Check file permissions

3. **Train model if missing:**
   ```bash
   python main.py
   ```

### Port Issues

If the app fails to start:

1. **Check PORT environment variable:**
   - Most platforms set this automatically
   - Verify in platform dashboard

2. **Check for port conflicts:**
   - Ensure no other service is using the port

### Memory Issues

If you encounter memory errors:

1. **Reduce gunicorn workers:**
   ```bash
   gunicorn --workers 1 --threads 2 app:app
   ```

2. **Increase platform memory limits** (if using cloud platform)

## Local Production Testing

Test production setup locally:

```bash
# Set production environment
export FLASK_DEBUG=false
export PORT=8080

# Run with gunicorn
gunicorn --bind 0.0.0.0:8080 --workers 2 --threads 2 app:app
```

## API Usage

Once deployed, use the API:

```python
import requests

# Example prediction
response = requests.post('https://your-app-url.com/api/predict', json={
    'material': 'Al',
    'ph': 12.0,
    'time_days': 1.0,
    'cement_type': 'CEM_I',
    'form_type': 'Concrete',
    'stat_measure': 'CL_Minus'
})

result = response.json()
print(f"Prediction: {result['prediction']} mg/m²")
```

## Support

For issues or questions:
1. Check the application logs in your deployment platform
2. Verify model files are present
3. Test the `/api/status` endpoint
4. Review the main README.md for general usage

