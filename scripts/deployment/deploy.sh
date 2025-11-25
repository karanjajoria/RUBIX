#!/bin/bash

# Deployment script for Google Cloud Run
# AI-Powered Refugee Crisis Intelligence System

set -e

echo "=========================================="
echo "Google Cloud Run Deployment Script"
echo "=========================================="

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
REGION="us-central1"
SERVICE_NAME="refugee-crisis-intelligence"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo ""
echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Service Name: $SERVICE_NAME"
echo "  Image: $IMAGE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Step 1: Authenticate with Google Cloud
echo "Step 1: Authenticating with Google Cloud..."
gcloud auth configure-docker

# Step 2: Build Docker image
echo ""
echo "Step 2: Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

# Step 3: Push image to Google Container Registry
echo ""
echo "Step 3: Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}:latest

# Step 4: Deploy to Cloud Run
echo ""
echo "Step 4: Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY}" \
  --project ${PROJECT_ID}

echo ""
echo "=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo ""
echo "Service URL:"
gcloud run services describe ${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --format 'value(status.url)'

echo ""
echo "To view logs:"
echo "  gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50"
echo ""
