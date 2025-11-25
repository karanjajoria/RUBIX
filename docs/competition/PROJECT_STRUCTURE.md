# Project Structure

Clean and organized file structure for the AI-Powered Refugee Crisis Intelligence System.

```
Google-Kaggle/
│
├── 📄 README.md                    # Main project documentation
├── 📄 START_HERE.md                # Quick start guide (begin here!)
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env                         # API keys (add your Gemini key here)
├── 📄 .gitignore                   # Git ignore rules
│
├── 🚀 main.py                      # Main application entry point
├── 🎓 train.py                     # Model training script
│
├── 📁 agents/                      # AI Agent implementations
│   ├── __init__.py
│   ├── vision_agent.py             # YOLO + Gemini vision intelligence
│   ├── forecasting_agent.py        # LSTM displacement forecasting
│   ├── resource_agent.py           # Resource optimization
│   ├── communication_agent.py      # Crisis communication
│   └── orchestrator_agent.py       # Multi-agent coordination
│
├── 📁 config/                      # Configuration files
│   ├── __init__.py
│   └── config.py                   # System configuration
│
├── 📁 utils/                       # Utility functions
│   ├── __init__.py
│   ├── memory.py                   # Memory management systems
│   └── training_data_prep.py       # Data preparation utilities
│
├── 📁 models/                      # Model files
│   ├── weights/                    # Pre-trained models
│   │   └── yolov8n.pt             # YOLO base model (6.3 MB)
│   ├── trained/                    # Your trained models
│   │   ├── lstm_forecaster.pth    # Trained LSTM (after training)
│   │   ├── scaler.pkl             # Feature scaler
│   │   ├── lstm_metadata.json     # Training metadata
│   │   └── yolo_conflict_custom.pt # Custom YOLO (optional)
│   └── README.md                   # Model documentation
│
├── 📁 data/                        # Datasets
│   ├── unhcr_refugees_processed.csv        # ✅ 693 rows
│   ├── acled_conflicts_processed.csv       # ✅ 2,566 rows
│   ├── worldbank_indicators.csv            # ✅ 30 rows
│   ├── climate_data.csv                    # ✅ 600 rows
│   │
│   ├── UNHCR Refugee Data/                 # Original UNHCR files
│   ├── ACLED Conflict Events/              # Original ACLED files
│   │
│   └── yolo_dataset/                       # YOLO training data (optional)
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
├── 📁 logs/                        # Application logs
│   └── episodic_memory.json       # Agent decision logs
│
├── 📁 docs/                        # Documentation
│   ├── DATA_SOURCES.md            # Data source information
│   ├── DATA_READY.md              # Your data summary
│   ├── DEPLOYMENT.md              # Google Cloud deployment guide
│   ├── GET_API_KEY.md             # How to get Gemini API key
│   ├── PROJECT_SUMMARY.md         # Competition strategy
│   ├── QUICKSTART.md              # Quick start guide
│   ├── QUICK_COMMANDS.md          # Command reference
│   ├── TRAINING_COMPLETE.md       # Training system summary
│   ├── TRAINING_GUIDE.md          # Complete training documentation
│   └── VIDEO_SCRIPT.md            # YouTube video script
│
├── 📁 scripts/                     # Utility scripts
│   ├── download_data.py            # Download datasets (original)
│   ├── download_remaining_data.py  # Download World Bank & Climate
│   ├── process_my_data.py          # Process your downloaded data
│   ├── install.bat                 # Windows installation script
│   ├── requirements_simplified.txt # Simplified dependencies
│   │
│   └── deployment/                 # Deployment files
│       ├── deploy.sh               # Deployment script
│       ├── cloudbuild.yaml         # Google Cloud Build config
│       ├── Dockerfile              # Docker container config
│       └── .dockerignore           # Docker ignore rules
│
└── 📁 env/                         # Python virtual environment (gitignored)
```

---

## 📂 Folder Purposes

### Core Application
- **agents/** - The 5 AI agents (vision, forecasting, resource, communication, orchestrator)
- **config/** - System configuration and settings
- **utils/** - Helper functions and utilities
- **main.py** - Run the demo or production system
- **train.py** - Train LSTM and YOLO models

### Data & Models
- **data/** - All datasets (UNHCR, ACLED, World Bank, Climate)
- **models/** - Pre-trained and trained models
- **logs/** - Application and memory logs

### Documentation
- **docs/** - All documentation files
- **README.md** - Main project overview
- **START_HERE.md** - Quick start guide (begin here!)

### Utilities
- **scripts/** - Data download and processing scripts
- **scripts/deployment/** - Deployment to Google Cloud

---

## 🎯 Most Important Files

### To Get Started:
1. **START_HERE.md** - Read this first!
2. **.env** - Add your Gemini API key here
3. **main.py** - Run the system
4. **train.py** - Train models

### For Training:
1. **train.py** - Main training script
2. **data/** - Your datasets (already processed!)
3. **docs/TRAINING_GUIDE.md** - Complete training docs

### For Understanding:
1. **README.md** - Project overview
2. **docs/PROJECT_SUMMARY.md** - Competition strategy
3. **agents/** - See how each agent works

---

## 🗑️ Can Be Deleted

None! All files are organized and useful. If you want to clean up:

- **env/** - Virtual environment (can recreate with `python -m venv env`)
- **logs/** - Log files (regenerated when running)
- **models/trained/** - Trained models (can retrain)

---

## 📊 File Sizes

```
Total project size: ~50 MB
├── models/weights/yolov8n.pt: 6.3 MB
├── data/: ~10 MB (your datasets)
├── env/: ~500 MB (Python packages - gitignored)
└── Rest: ~5 MB (code + docs)
```

---

## 🚀 Quick Navigation

**Want to...**
- **Get started?** → [START_HERE.md](START_HERE.md)
- **Train models?** → `python train.py --model lstm --epochs 100`
- **Run demo?** → `python main.py --mode demo`
- **Deploy?** → [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Understand data?** → [docs/DATA_READY.md](docs/DATA_READY.md)
- **Get API key?** → [docs/GET_API_KEY.md](docs/GET_API_KEY.md)
- **See commands?** → [docs/QUICK_COMMANDS.md](docs/QUICK_COMMANDS.md)

---

**Last Updated**: November 23, 2024
