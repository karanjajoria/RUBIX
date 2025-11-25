# Model Training Guide

Complete guide for training custom LSTM and YOLO models for the Refugee Crisis Intelligence System.

## Table of Contents

1. [Quick Start](#quick-start)
2. [LSTM Training](#lstm-training)
3. [YOLO Training](#yolo-training)
4. [Data Preparation](#data-preparation)
5. [Using Trained Models](#using-trained-models)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Download training data
python download_data.py
```

### Train All Models (Recommended)

```bash
# Train both LSTM and YOLO models
python train.py --model all --epochs 50
```

### Train Individual Models

```bash
# LSTM only (displacement forecasting)
python train.py --model lstm --epochs 100 --batch-size 32 --learning-rate 0.001

# YOLO only (conflict detection)
python train.py --model yolo --epochs 50
```

---

## LSTM Training

### What is it?

The LSTM model predicts refugee displacement based on 90+ features including:
- Conflict intensity
- Economic indicators (GDP, unemployment, inflation)
- Climate data (temperature, precipitation)
- Demographic factors
- Infrastructure metrics

### Training Process

**1. Prepare Data**

```bash
# Option A: Use downloaded real data
python download_data.py

# Option B: Prepare custom training data
python utils/training_data_prep.py
```

**2. Train Model**

```bash
python train.py --model lstm --epochs 100
```

**3. Monitor Training**

```
Epoch [10/100] - Train Loss: 0.0234, Val Loss: 0.0312
Epoch [20/100] - Train Loss: 0.0198, Val Loss: 0.0289
...
```

**4. Results**

- Trained model: `models/trained/lstm_forecaster.pth`
- Feature scaler: `models/trained/scaler.pkl`
- Metadata: `models/trained/lstm_metadata.json`

### Training Data Format

The LSTM expects time series data with these features:

```python
{
    'conflict': [
        'conflict_intensity', 'battle_deaths', 'violence_against_civilians',
        'political_instability_index', 'neighboring_conflicts'
    ],
    'economic': [
        'gdp_growth', 'unemployment_rate', 'inflation_rate',
        'food_insecurity_rate', 'poverty_rate', ...
    ],
    'climate': [
        'temperature_celsius', 'precipitation_mm', 'drought_index', ...
    ],
    # ... and 90+ more features
}
```

### Hyperparameters

```bash
python train.py --model lstm \
    --epochs 100 \          # Number of training epochs
    --batch-size 32 \       # Batch size
    --learning-rate 0.001   # Learning rate
```

**Default Architecture:**
- Hidden size: 128
- Num layers: 2
- Dropout: 0.2
- Optimizer: Adam
- Loss: MSE

### Performance Metrics

The training script reports:
- **Train Loss**: MSE on training set
- **Validation Loss**: MSE on validation set (20% holdout)
- **Best Model**: Automatically saved when validation loss improves

---

## YOLO Training

### What is it?

The YOLO model detects conflict-related objects in satellite imagery:
- Refugee camps
- Military vehicles
- Destroyed buildings
- Displaced populations
- Aid convoys

### Training Process

**1. Prepare Dataset**

YOLO requires labeled satellite images in a specific format:

```
data/yolo_dataset/
├── images/
│   ├── train/          # Training images (.jpg, .png)
│   └── val/            # Validation images
├── labels/
│   ├── train/          # Annotations (.txt, YOLO format)
│   └── val/
└── data.yaml           # Dataset configuration
```

**YOLO Annotation Format:**

Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.3         # refugee_camp at center
1 0.2 0.3 0.1 0.1         # military_vehicle
```

Values are normalized (0-1 relative to image dimensions).

**2. Label Your Data**

Use labeling tools:
- [LabelImg](https://github.com/HumanSignal/labelImg) (free, local)
- [Roboflow](https://roboflow.com/) (online, free tier)
- [CVAT](https://www.cvat.ai/) (free, self-hosted)

**3. Train Model**

```bash
python train.py --model yolo --epochs 50
```

**4. Results**

- Trained model: `models/trained/yolo_conflict_custom.pt`
- Training outputs: `models/trained/refugee_crisis_detector/`
- Metadata: `models/trained/yolo_metadata.json`

### Classes

```yaml
0: refugee_camp           # Tent clusters, temporary settlements
1: military_vehicle       # Tanks, armored vehicles
2: destroyed_building     # Damaged infrastructure
3: displaced_population   # Large gatherings of people
4: aid_convoy            # UN trucks, supply vehicles
```

### Data Requirements

**Minimum for training:**
- 100+ labeled images per class (train)
- 30+ labeled images per class (validation)

**Recommended:**
- 500+ labeled images per class
- Diverse conditions (day/night, weather, regions)

**Data Sources:**

1. **Satellite Imagery:**
   - [Sentinel Hub](https://www.sentinel-hub.com/) (free, EU Copernicus)
   - [Planet Labs](https://www.planet.com/) (commercial)
   - [Maxar Open Data](https://www.maxar.com/open-data) (crisis events)

2. **Conflict Region Databases:**
   - ACLED (crisis event locations)
   - UNHCR (refugee camp coordinates)

### Hyperparameters

```bash
python train.py --model yolo --epochs 50
```

**Default Configuration:**
- Model: YOLOv8n (nano - fast, 6.3 MB)
- Image size: 640x640
- Batch size: 16
- Patience: 20 epochs (early stopping)

**Upgrade to YOLOv8m or YOLOv8x for higher accuracy** (at cost of speed/size).

---

## Data Preparation

### Automated Preparation

```bash
# Run the data preparation utility
python utils/training_data_prep.py
```

This will:
1. Merge all downloaded data sources (UNHCR, World Bank, Climate, ACLED)
2. Engineer features matching LSTM requirements
3. Augment time series if data is limited
4. Export to `data/lstm_training_data.csv`
5. Create sample YOLO annotations
6. Validate all data

### Manual Preparation

**For LSTM:**

```python
from utils.training_data_prep import merge_all_data_sources, create_lstm_features

# Merge sources
df = merge_all_data_sources()

# Engineer features
df_features = create_lstm_features(df)

# Export
df_features.to_csv('data/lstm_training_data.csv', index=False)
```

**For YOLO:**

1. Collect satellite images of refugee crises
2. Label using LabelImg/Roboflow
3. Export in YOLO format
4. Place in `data/yolo_dataset/`

### Data Validation

```bash
# Check if data is ready for training
python utils/training_data_prep.py
```

---

## Using Trained Models

### Automatic Integration

Once trained, the agents automatically use trained models:

```python
# Vision Agent
# Automatically loads: models/trained/yolo_conflict_custom.pt (if exists)
# Falls back to: models/weights/yolov8n.pt

# Forecasting Agent
# Automatically loads: models/trained/lstm_forecaster.pth (if exists)
# Falls back to: untrained LSTM model
```

### Running Demo with Trained Models

```bash
# Add Gemini API key first
# (see GET_API_KEY.md)

# Run demo
python main.py --mode demo
```

You'll see:
```
[Vision Agent] Loaded TRAINED YOLO model: models/trained/yolo_conflict_custom.pt
[Forecasting Agent] Loaded trained LSTM model from models/trained/lstm_forecaster.pth
[Forecasting Agent] Loaded trained scaler from models/trained/scaler.pkl
```

### Manual Loading (for testing)

```python
import torch
from ultralytics import YOLO

# Load LSTM
model = LSTMForecaster(input_size=20)
model.load_state_dict(torch.load('models/trained/lstm_forecaster.pth'))
model.eval()

# Load YOLO
yolo = YOLO('models/trained/yolo_conflict_custom.pt')
results = yolo('satellite_image.jpg')
```

---

## Troubleshooting

### LSTM Training Issues

**Issue: "No data found"**
```bash
# Solution: Download data first
python download_data.py
```

**Issue: "CUDA out of memory"**
```bash
# Solution: Reduce batch size
python train.py --model lstm --batch-size 16
```

**Issue: "High validation loss"**
- Increase epochs: `--epochs 200`
- Add more training data
- Check for data quality issues

### YOLO Training Issues

**Issue: "No training images found"**
```
# Solution: YOLO needs labeled images
# Add images to: data/yolo_dataset/images/train/
# Add labels to: data/yolo_dataset/labels/train/
```

**Issue: "ultralytics not installed"**
```bash
pip install ultralytics
```

**Issue: Training very slow**
- Use GPU: Ensure CUDA is installed
- Use smaller model: YOLOv8n (default)
- Reduce image size: `--img-size 320`

### General Issues

**Issue: "Import errors"**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Issue: "Permission denied" (Windows)**
- Run terminal as Administrator
- Check antivirus isn't blocking files

**Issue: "Out of memory"**
- Close other applications
- Reduce batch size
- Use CPU instead of GPU (slower but works)

---

## Training Metrics & Benchmarks

### Expected LSTM Performance

With 120+ months of data:
- Train Loss: < 0.02 (good)
- Val Loss: < 0.03 (good)
- Training time: ~5-10 minutes on CPU

### Expected YOLO Performance

With 500+ labeled images:
- mAP@0.5: > 0.6 (decent)
- mAP@0.5:0.95: > 0.4 (decent)
- Training time: ~1-2 hours on CPU, ~15 min on GPU

---

## Next Steps After Training

1. **Validate Performance:**
   ```bash
   python main.py --mode demo
   ```

2. **Deploy to Production:**
   ```bash
   ./deploy.sh
   ```
   (See [DEPLOYMENT.md](DEPLOYMENT.md))

3. **Monitor in Production:**
   - Track prediction accuracy
   - Use conversation memory to compare predictions vs actuals
   - Retrain periodically with new data

4. **Iterate:**
   - Collect more labeled data
   - Tune hyperparameters
   - Experiment with larger models (YOLOv8m, deeper LSTM)

---

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [LabelImg Tool](https://github.com/HumanSignal/labelImg)
- [UNHCR Data Portal](https://data.unhcr.org/)
- [ACLED Conflict Data](https://acleddata.com/)

---

**Questions?** Check the main [README.md](README.md) or open an issue.
