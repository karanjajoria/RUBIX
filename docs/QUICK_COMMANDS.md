# Quick Command Reference

## Training Commands

```bash
# Train LSTM (displacement forecasting)
python train.py --model lstm --epochs 100

# Train YOLO (conflict detection - needs labeled images)
python train.py --model yolo --epochs 50

# Train both models
python train.py --model all --epochs 50

# Quick test (5 epochs)
python train.py --model lstm --epochs 5
```

## Data Commands

```bash
# Download real refugee crisis data
python download_data.py

# Prepare and validate training data
python utils/training_data_prep.py
```

## Running the System

```bash
# Run demo (requires Gemini API key in .env)
python main.py --mode demo

# Run production mode
python main.py --mode production
```

## Model Management

```bash
# Check trained models
ls models/trained/

# View LSTM metadata
cat models/trained/lstm_metadata.json

# View training results
cat models/trained/training_results.json
```

## Setup Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Or use simplified requirements
pip install -r requirements_simplified.txt

# Windows batch install
install.bat
```

## Deployment

```bash
# Deploy to Google Cloud Run
./deploy.sh

# Build Docker image
docker build -t refugee-crisis-ai .
```

## Useful Paths

- **Trained models**: `models/trained/`
- **Data**: `data/`
- **Logs**: `logs/`
- **Config**: `config/config.py`
- **Agents**: `agents/`

## Training Parameters

```bash
# Custom LSTM training
python train.py --model lstm \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 0.0005
```

## Documentation Files

- `README.md` - Project overview
- `QUICKSTART.md` - Quick start guide
- `TRAINING_GUIDE.md` - Complete training docs
- `DEPLOYMENT.md` - Deployment guide
- `GET_API_KEY.md` - API setup
- `TRAINING_COMPLETE.md` - Training summary
- `VIDEO_SCRIPT.md` - Demo video script
- `PROJECT_SUMMARY.md` - Competition strategy

## Check System Status

```bash
# Test imports
python -c "from agents.forecasting_agent import forecasting_agent; print('OK')"

# Check YOLO model
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Validate data
python utils/training_data_prep.py
```
