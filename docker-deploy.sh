#!/bin/bash
# Docker deployment script for Cement Leaching Prediction App

set -e

echo "🐳 Docker Deployment Script"
echo "=========================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if model files exist
if [ ! -f "models/production_model.pkl" ]; then
    echo "⚠️  Warning: Model files not found in models/ directory"
    echo "   Please train the model first by running: python main.py"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t cement-leaching-app:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo ""
    echo "🚀 Starting container..."
    echo "   App will be available at: http://localhost:8080"
    echo "   API endpoint: http://localhost:8080/api/predict"
    echo ""
    echo "   Press Ctrl+C to stop the container"
    echo ""
    
    # Run the container
    docker run -p 8080:8080 --name cement-leaching-app cement-leaching-app:latest
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

