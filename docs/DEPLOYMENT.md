# Deployment Guide
## AI-Powered Refugee Crisis Intelligence System

This guide covers deployment to Google Cloud Run for scalable, production-ready operation.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
3. **Docker** installed ([Install Guide](https://docs.docker.com/get-docker/))
4. **Gemini API Key** from Google AI Studio
5. **Project created** in Google Cloud Console

## Quick Deployment (Automated)

### Option 1: Using Deployment Script

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GEMINI_API_KEY="your-gemini-api-key"

# Make script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### Option 2: Using Cloud Build (CI/CD)

```bash
# Submit build to Google Cloud Build
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _GEMINI_API_KEY="your-gemini-api-key"
```

## Manual Deployment (Step-by-Step)

### Step 1: Initialize Google Cloud

```bash
# Authenticate with Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Step 2: Build Docker Image

```bash
# Configure Docker to use gcloud credentials
gcloud auth configure-docker

# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/refugee-crisis-intelligence:latest .

# Test locally (optional)
docker run -p 8080:8080 \
  -e GEMINI_API_KEY="your-key" \
  gcr.io/YOUR_PROJECT_ID/refugee-crisis-intelligence:latest
```

### Step 3: Push to Container Registry

```bash
# Push image to GCR
docker push gcr.io/YOUR_PROJECT_ID/refugee-crisis-intelligence:latest
```

### Step 4: Deploy to Cloud Run

```bash
# Deploy with environment variables
gcloud run deploy refugee-crisis-intelligence \
  --image gcr.io/YOUR_PROJECT_ID/refugee-crisis-intelligence:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --set-env-vars "GEMINI_API_KEY=your-key"
```

### Step 5: Verify Deployment

```bash
# Get service URL
gcloud run services describe refugee-crisis-intelligence \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'

# View logs
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=refugee-crisis-intelligence" \
  --limit 50
```

## Configuration

### Environment Variables

Set these in Cloud Run:

```bash
GEMINI_API_KEY=your-gemini-api-key           # Required
TWILIO_ACCOUNT_SID=your-twilio-sid           # Optional (for SMS)
TWILIO_AUTH_TOKEN=your-twilio-token          # Optional
TWILIO_PHONE_NUMBER=your-twilio-number       # Optional
LOG_LEVEL=INFO                               # Optional (default: INFO)
DEBUG_MODE=False                             # Optional (default: False)
```

Update environment variables:

```bash
gcloud run services update refugee-crisis-intelligence \
  --region us-central1 \
  --set-env-vars "GEMINI_API_KEY=new-key,LOG_LEVEL=DEBUG"
```

### Resource Limits

Default configuration:
- **Memory**: 2 GiB (handles YOLO + LSTM models)
- **CPU**: 2 vCPUs (parallel agent execution)
- **Timeout**: 900 seconds (15 minutes for complex workflows)

Adjust if needed:

```bash
gcloud run services update refugee-crisis-intelligence \
  --region us-central1 \
  --memory 4Gi \
  --cpu 4
```

## Cost Optimization

### Pricing Estimates (us-central1)

**Cloud Run** (per month with 1000 requests/day):
- CPU: ~$0.024 per vCPU-second
- Memory: ~$0.0025 per GiB-second
- Requests: First 2M free, then $0.40 per million
- **Estimated**: ~$50-100/month

**Container Registry**:
- Storage: $0.026 per GB/month
- **Estimated**: ~$1-5/month

**Gemini API** (1000 requests/day):
- Gemini 2.0 Flash: ~$0.075 per 1M input tokens
- **Estimated**: ~$20-50/month

**Total Estimated Cost**: ~$70-155/month

### Cost Reduction Tips

1. **Use Gemini Flash** for high-frequency tasks (already configured)
2. **Set min instances to 0** (default) to scale to zero when idle
3. **Implement caching** for repeated satellite image analyses
4. **Use Cloud Storage** for large datasets instead of including in container
5. **Set up budget alerts** in Google Cloud Console

## Alternative Deployment: Agent Engine

For native multi-agent deployment using Google's Agent Engine:

```bash
# Install ADK (Agent Development Kit)
pip install google-cloud-aiplatform

# Deploy with ADK
adk deploy agent_engine \
  --project=YOUR_PROJECT_ID \
  --region=us-central1 \
  --staging_bucket=gs://your-bucket \
  --display_name="Refugee Crisis Intelligence" \
  /path/to/agent/directory
```

## Monitoring & Logging

### View Logs

```bash
# Real-time logs
gcloud logging tail \
  "resource.type=cloud_run_revision AND resource.labels.service_name=refugee-crisis-intelligence"

# Filter by severity
gcloud logging read \
  "resource.type=cloud_run_revision AND severity>=ERROR" \
  --limit 50
```

### Set Up Alerts

1. Go to **Google Cloud Console** → **Monitoring** → **Alerting**
2. Create alert for:
   - Error rate > 5%
   - CPU usage > 80%
   - Memory usage > 90%
   - Request latency > 30s

### Performance Metrics

Monitor in Cloud Console:
- **Request count**: Track usage patterns
- **Request latency**: Optimize slow agents
- **Container CPU/Memory**: Right-size resources
- **Error rate**: Debug agent failures

## Scaling Configuration

### Autoscaling

Cloud Run automatically scales based on:
- Concurrent requests per instance
- CPU/memory utilization
- Request latency

Configure autoscaling:

```bash
gcloud run services update refugee-crisis-intelligence \
  --region us-central1 \
  --min-instances 0 \
  --max-instances 100 \
  --concurrency 80
```

### Load Testing

Test scaling with Apache Bench:

```bash
ab -n 1000 -c 10 https://your-service-url.run.app/
```

## Security

### API Key Management

**Never commit API keys!** Use Secret Manager:

```bash
# Create secret
echo -n "your-gemini-api-key" | \
  gcloud secrets create gemini-api-key --data-file=-

# Grant Cloud Run access
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member=serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor

# Update Cloud Run to use secret
gcloud run services update refugee-crisis-intelligence \
  --region us-central1 \
  --update-secrets GEMINI_API_KEY=gemini-api-key:latest
```

### Network Security

- **VPC Connector**: Connect to private resources
- **IAM**: Restrict access to authorized users
- **HTTPS**: Always enabled by default

## Troubleshooting

### Common Issues

**1. Container fails to start**
```bash
# Check logs
gcloud logging read "resource.labels.service_name=refugee-crisis-intelligence" --limit 50

# Test locally
docker run -it gcr.io/YOUR_PROJECT_ID/refugee-crisis-intelligence:latest
```

**2. Out of memory**
```bash
# Increase memory
gcloud run services update refugee-crisis-intelligence \
  --region us-central1 \
  --memory 4Gi
```

**3. Timeout errors**
```bash
# Increase timeout
gcloud run services update refugee-crisis-intelligence \
  --region us-central1 \
  --timeout 1800  # 30 minutes
```

**4. Gemini API errors**
- Verify API key is set correctly
- Check quota limits in Google AI Studio
- Review rate limiting settings

### Debug Mode

Enable debug logging:

```bash
gcloud run services update refugee-crisis-intelligence \
  --region us-central1 \
  --set-env-vars "DEBUG_MODE=True,LOG_LEVEL=DEBUG"
```

## Continuous Deployment

### GitHub Actions Integration

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Cloud SDK
        uses: google-github-actions/setup-gcloud@v0
        with:
          project_id: ${{ secrets.GCP_PROJECT_ID }}
          service_account_key: ${{ secrets.GCP_SA_KEY }}

      - name: Build and Deploy
        run: |
          gcloud builds submit --config cloudbuild.yaml
```

## Production Checklist

Before going to production:

- [ ] Set up monitoring alerts
- [ ] Configure backup for episodic memory logs
- [ ] Implement rate limiting
- [ ] Set up Cloud Storage for satellite images
- [ ] Train YOLO model on real conflict imagery
- [ ] Train LSTM on real UNHCR data
- [ ] Configure VPC for data security
- [ ] Set up budget alerts
- [ ] Document runbooks for incident response
- [ ] Test disaster recovery procedures

## Support

For deployment issues:
- Check [Google Cloud Run documentation](https://cloud.google.com/run/docs)
- Review [Gemini API documentation](https://ai.google.dev/docs)
- Open an issue in the project repository

---

**Last Updated**: 2025-01-23
