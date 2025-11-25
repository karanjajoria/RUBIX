# Quick Start Guide
## Get Your Project Running in 5 Minutes

This guide will get you up and running with the AI-Powered Refugee Crisis Intelligence System.

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd c:\Users\KARAN\Desktop\College\projects\Google-Kaggle

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables (1 minute)

```bash
# Copy example environment file
copy .env.example .env

# Edit .env file and add your Gemini API key
# Get your key from: https://aistudio.google.com/app/apikey
```

In `.env`, add:
```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

## Step 3: Run the Demo (2 minutes)

```bash
# Run demo mode
python main.py --mode demo
```

You should see:
```
================================================================================
AI-POWERED REFUGEE CRISIS INTELLIGENCE SYSTEM - DEMO MODE
================================================================================

--------------------------------------------------------------------------------
DEMO 1: PARALLEL WORKFLOW (Vision + Forecasting simultaneously)
--------------------------------------------------------------------------------

[Orchestrator] Starting PARALLEL workflow: Vision + Forecasting
...
[Parallel Workflow Results]
Vision Threat Level: high
Vision Threat Score: 8.5/10
Predicted Displacement: 2,847 people
Execution Time: 12.34s
Validation Passed: True
...
```

## What You'll See

The demo runs three workflows:

### 1. Parallel Workflow (15-20 seconds)
- Vision Agent analyzes satellite image
- Forecasting Agent processes historical data
- Both run simultaneously
- Results validated by Orchestrator

### 2. Sequential Workflow (30-40 seconds)
- Vision → Forecasting → Resource → Communication
- Shows complete pipeline
- Sends mock SMS alerts

### 3. Looped Workflow (40-50 seconds)
- 3 iterations of refinement
- Shows how threat predictions improve
- Demonstrates continuous learning

### 4. Memory Summary
- Shows stored predictions, episodes, and embeddings

## Troubleshooting

### Error: "GEMINI_API_KEY not found"
**Solution**: Make sure you created `.env` file and added your API key.

### Error: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Run `pip install -r requirements.txt` again.

### Error: "OpenCV not installed"
**Solution**: The system will use mock detections. This is normal for demo mode.

### Warning: "Twilio not configured"
**Solution**: This is normal. SMS features work in mock mode for demo.

## Next Steps

### To Customize the Demo

Edit [main.py:175-185](main.py#L175-L185) to change:
- Location coordinates
- Number of loop iterations
- Alert recipients

### To Run with Real Data

1. Create a CSV file with historical data (see format in code comments)
2. Add satellite images to `data/sample/`
3. Run: `python main.py --mode full --data_path your_data.csv`

### To Deploy to Google Cloud

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide.

Quick deploy:
```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Run deployment script
chmod +x deploy.sh
./deploy.sh
```

## Project Structure Reference

```
Google-Kaggle/
├── agents/                 # 5 agent implementations
│   ├── vision_agent.py         # YOLO + Gemini vision
│   ├── forecasting_agent.py    # LSTM forecasting
│   ├── resource_agent.py       # Resource optimization
│   ├── communication_agent.py  # Multi-lingual alerts
│   └── orchestrator_agent.py   # Workflow coordination
│
├── config/                 # Configuration
├── utils/                  # Memory management
├── data/                   # Data storage
├── logs/                   # Episode logs
│
├── main.py                # ← START HERE
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
└── README.md             # Full documentation
```

## Understanding the Output

### Vision Agent Output
```
Threat Level: high          # Categorical risk (low/medium/high/critical)
Threat Score: 8.5/10       # Numeric score for severity
Detections: 3              # Number of threats detected
```

### Forecasting Agent Output
```
Predicted Displacement: 2,847 people    # Total expected displacement
Forecast Horizon: 6 months              # Prediction timeframe
Lead Time: 4.2 months                   # Warning time
```

### Resource Agent Output
```
Water Points: 12           # Calculated based on population
Health Centers: 2          # Infrastructure needs
Priority: high            # Urgency level
```

### Communication Agent Output
```
Alerts Sent: 4             # Number of messages delivered
Languages: en, fr          # Multi-lingual support
Delivery Status: [...]     # Per-recipient confirmation
```

## Common Questions

**Q: How long does the demo take?**
A: About 90-120 seconds total for all three workflows.

**Q: Do I need real satellite images?**
A: No. The demo creates placeholder images automatically.

**Q: Will this use a lot of API credits?**
A: No. Demo uses ~$0.05 worth of Gemini API calls.

**Q: Can I run this offline?**
A: No. Requires internet for Gemini API calls.

**Q: Is the data real?**
A: The demo uses synthetic data modeled after real Uganda refugee patterns.

## Getting Help

- **Installation issues**: Check `requirements.txt` versions
- **API errors**: Verify Gemini API key is valid
- **Code questions**: See inline comments in each agent file
- **Deployment help**: See [DEPLOYMENT.md](DEPLOYMENT.md)

## What to Do After Demo

1. ✅ Review the code in each agent file
2. ✅ Read [README.md](README.md) for full documentation
3. ✅ Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for competition strategy
4. ✅ Read [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) for video production
5. ✅ Deploy to Google Cloud using [DEPLOYMENT.md](DEPLOYMENT.md)

---

**Ready to compete? The system is fully functional and competition-ready!**

For questions or issues, check the full documentation in README.md or open an issue in the repository.
