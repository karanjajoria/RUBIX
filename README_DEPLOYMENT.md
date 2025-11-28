# Deployment Guide
## AI-Powered Refugee Crisis Intelligence System

This guide shows you how to deploy your project online so others can use it.

---

## 🚀 Option 1: Hugging Face Spaces (RECOMMENDED - FREE)

**Best for:** AI/ML projects with interactive demos
**Cost:** FREE
**Features:** GPU support, gradio interface, easy deployment

### Step-by-Step Instructions:

#### 1. Create Hugging Face Account
- Go to [huggingface.co](https://huggingface.co)
- Click "Sign Up" (it's free)
- Verify your email

#### 2. Create a New Space
- Click your profile → "New Space"
- Fill in details:
  - **Name:** `refugee-crisis-ai`
  - **License:** MIT
  - **SDK:** Gradio
  - **Hardware:** CPU Basic (free) or GPU (if available)
- Click "Create Space"

#### 3. Upload Files
You can upload via web interface or Git:

**Option A: Web Upload (Easier)**
1. Click "Files" tab in your Space
2. Click "Add file" → "Upload files"
3. Upload these files:
   - `app.py` (main Gradio app)
   - `requirements_huggingface.txt` (rename to `requirements.txt`)
   - `agents/` folder (all agent files)
   - `models/trained/` folder (LSTM model files)
   - `config/` folder
   - `utils/` folder
   - `data/sample/` folder (sample images)

**Option B: Git Upload (Advanced)**
```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/refugee-crisis-ai
cd refugee-crisis-ai

# Copy files
cp app.py README.md requirements_huggingface.txt .
cp -r agents models config utils data .

# Rename requirements file
mv requirements_huggingface.txt requirements.txt

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

#### 4. Set Environment Variables (Optional)
If using Gemini API:
1. Go to Space Settings → "Variables and secrets"
2. Add secret: `GEMINI_API_KEY` = your API key
3. Save

#### 5. Wait for Build
- Your Space will automatically build (takes 5-10 minutes)
- You'll see build logs in the "Logs" tab
- Once done, your app will be live!

#### 6. Share Your Space
Your app will be at: `https://huggingface.co/spaces/YOUR_USERNAME/refugee-crisis-ai`

### 🎨 Customize Your Space

Edit the `README.md` in your Space:
```markdown
---
title: Refugee Crisis Intelligence AI
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
---

# AI-Powered Refugee Crisis Intelligence System

Multi-agent AI system predicting refugee displacement 4-6 months ahead.

## Features
- 5 specialized AI agents
- LSTM neural network (87% accuracy)
- Real-time crisis prediction
- Interactive web interface

## How to Use
1. Select a region from the dropdown
2. Click "Run Analysis"
3. View predictions and recommendations

Built for Google Kaggle "Agents for Good" Competition | Team RUBIX
```

---

## 🌐 Option 2: Render.com (FREE Tier)

**Best for:** Full-stack deployment with backend
**Cost:** FREE (with limitations)

### Quick Deploy:

#### 1. Create Account
- Go to [render.com](https://render.com)
- Sign up with GitHub

#### 2. Create Web Service
- Click "New +" → "Web Service"
- Connect your GitHub repo
- Configure:
  - **Name:** `refugee-crisis-ai`
  - **Environment:** Python 3
  - **Build Command:** `pip install -r requirements.txt`
  - **Start Command:** `python web/app.py`
  - **Plan:** Free

#### 3. Environment Variables
Add in Render dashboard:
```
GEMINI_API_KEY=your_key_here
PORT=5000
```

#### 4. Deploy
- Click "Create Web Service"
- Wait 10-15 minutes for deployment
- Your app will be at: `https://refugee-crisis-ai.onrender.com`

**Note:** Free tier sleeps after 15 min of inactivity (takes 30s to wake up).

---

## ☁️ Option 3: Google Cloud Run (FREE $300 Credit)

**Best for:** Production deployment
**Cost:** FREE credit for new users, then pay-as-you-go

### Quick Deploy:

```bash
# 1. Install Google Cloud SDK
# Windows: Download from https://cloud.google.com/sdk/docs/install
# Mac: brew install google-cloud-sdk
# Linux: curl https://sdk.cloud.google.com | bash

# 2. Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/refugee-crisis-ai
gcloud run deploy refugee-crisis-ai \
  --image gcr.io/YOUR_PROJECT_ID/refugee-crisis-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key
```

Your app will be at: `https://refugee-crisis-ai-xxxxx.run.app`

---

## 🐳 Option 4: Railway.app (FREE $5/month credit)

**Best for:** Quick deployment with minimal config
**Cost:** $5 free credit/month

### Quick Deploy:

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Python and deploys
5. Add environment variables in dashboard
6. Your app is live!

---

## 📦 Option 5: Streamlit Cloud (FREE for public repos)

**Best for:** Data science/ML demos

### Convert to Streamlit:

Create `streamlit_app.py`:
```python
import streamlit as st
from agents.orchestrator_agent import OrchestratorAgent

st.set_page_config(page_title="Refugee Crisis AI", page_icon="🌍")

st.title("🌍 Refugee Crisis Intelligence System")
st.markdown("Multi-agent AI predicting displacement 4-6 months ahead")

region = st.selectbox("Select Region", [
    "Syria-Turkey Border",
    "Sudan-Chad Border",
    "Afghanistan-Pakistan Border"
])

if st.button("🚀 Run Analysis"):
    with st.spinner("Running multi-agent analysis..."):
        # Your analysis code here
        st.success("Analysis complete!")
```

Deploy:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub
3. Select repo, branch, and `streamlit_app.py`
4. Click "Deploy"

---

## 🎯 Recommended Deployment Strategy

### For Competition Judges:
**Use Hugging Face Spaces** ✅
- Free forever
- Easy to use interface
- GPU support if needed
- Professional demo URL
- No sleep/wake delays

### For Real Production:
**Use Google Cloud Run**
- Scalable
- Pay only for usage
- Production-grade
- Custom domain support

---

## 📊 What Gets Deployed

### Included in Deployment:
✅ All 5 AI agents (Python code)
✅ LSTM trained model weights
✅ Web interface (Gradio or Flask)
✅ Sample data for demos
✅ Documentation

### NOT Included (too large):
❌ Full training datasets (30 years of data)
❌ Raw satellite imagery
❌ Development files (.venv, __pycache__)

---

## 🔒 Security Notes

### Environment Variables:
**Never commit these to Git:**
- `GEMINI_API_KEY`
- `TWILIO_AUTH_TOKEN`
- Any API keys or secrets

**Always use:**
- `.env` file locally (in .gitignore)
- Platform environment variables for deployment

### Example .env.example:
```bash
GEMINI_API_KEY=your_key_here
TWILIO_ACCOUNT_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_token_here
```

---

## 🐛 Troubleshooting

### Common Issues:

**1. "Module not found" error**
- Check `requirements.txt` includes all dependencies
- Ensure file paths are correct (use relative paths)

**2. "Model file not found"**
- Verify model files are uploaded to deployment
- Check file paths in code

**3. "Memory error"**
- Upgrade to paid tier with more RAM
- Reduce model size or use model quantization

**4. "API key not working"**
- Check environment variable name matches code
- Verify key is valid and has quota

---

## 📈 Monitoring Your Deployment

### Hugging Face Spaces:
- View usage stats in Space dashboard
- Check logs in "Logs" tab
- Monitor GPU/CPU usage

### Render/Railway:
- Check logs in dashboard
- Set up alerts for errors
- Monitor response times

---

## 🔗 Share Your Deployment

### For Competition Submission:
```markdown
🌐 Live Demo: https://huggingface.co/spaces/YOUR_USERNAME/refugee-crisis-ai
📂 GitHub: https://github.com/YOUR_USERNAME/refugee-crisis-ai
📖 Documentation: https://github.com/YOUR_USERNAME/refugee-crisis-ai/docs
🎥 Video Demo: [YouTube Link]
```

### Social Media Post:
```
🌍 Just deployed my AI system for predicting refugee crises!

Try it live: [Your URL]

🤖 5 AI agents working together
📊 87% prediction accuracy
⚡ 4-6 month early warning

Built for @GoogleAI Kaggle competition 🏆

#AI #MachineLearning #HumanitarianTech #AIForGood
```

---

## ✅ Deployment Checklist

Before deploying:
- [ ] Test locally (python app.py)
- [ ] Update requirements.txt
- [ ] Remove sensitive data
- [ ] Add .env.example
- [ ] Update README with deployment URL
- [ ] Test with sample data
- [ ] Add error handling
- [ ] Set up environment variables
- [ ] Deploy to platform
- [ ] Test live deployment
- [ ] Share with competition judges

---

## 🎓 Need Help?

**Hugging Face Docs:** https://huggingface.co/docs/hub/spaces
**Render Docs:** https://render.com/docs
**Railway Docs:** https://docs.railway.app
**Google Cloud Docs:** https://cloud.google.com/run/docs

**Questions?** Open an issue on GitHub or contact Team RUBIX.

---

**Ready to deploy? Start with Hugging Face Spaces - it's the easiest! 🚀**
