# Models Directory

This directory stores all trained models and weights used by the multi-agent system.

## Structure

```
models/
├── weights/              # Pre-trained model weights
│   └── yolov8n.pt       # YOLO object detection model (6.3 MB)
│
├── trained/             # Custom trained models (for production)
│   ├── lstm_forecaster.pth      # LSTM displacement forecasting model
│   ├── yolo_conflict_custom.pt  # Fine-tuned YOLO on conflict imagery
│   └── scaler.pkl               # Feature scaler for LSTM
│
└── checkpoints/         # Training checkpoints (optional)
    └── ...
```

## Current Models

### 1. YOLOv8n (Object Detection)
- **File**: `weights/yolov8n.pt`
- **Size**: 6.3 MB
- **Purpose**: Detects objects in satellite imagery (weapons, military vehicles, etc.)
- **Status**: ✅ Downloaded (pretrained on COCO dataset)
- **Usage**: Used by Vision Intelligence Agent for threat detection

**For Production:**
- Fine-tune on conflict-specific imagery (weapons, military vehicles, damaged buildings)
- Dataset: Custom labeled satellite images or use open datasets like xView

### 2. LSTM Forecaster (Displacement Prediction)
- **File**: `trained/lstm_forecaster.pth` (not yet created)
- **Purpose**: Predicts refugee displacement 4-6 months ahead
- **Status**: ⏳ Uses mock predictions (model architecture ready in code)
- **Usage**: Used by Forecasting Agent

**To Train:**
```python
# Train LSTM on real UNHCR data
from agents.forecasting_agent import forecasting_agent
# Training code would go here
```

### 3. Feature Scaler
- **File**: `trained/scaler.pkl` (not yet created)
- **Purpose**: Normalizes 90+ input features for LSTM
- **Status**: ⏳ Created during LSTM training
- **Usage**: Used by Forecasting Agent to preprocess features

## Model Downloads

Models are automatically downloaded when first used:

1. **YOLO** - Downloads from Ultralytics on first run
2. **LSTM** - Needs to be trained on real data (or use mock predictions)

## Training Your Own Models

### Option 1: Fine-tune YOLO for Conflict Detection

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('weights/yolov8n.pt')

# Train on custom conflict imagery dataset
results = model.train(
    data='conflict_data.yaml',  # Dataset config
    epochs=100,
    imgsz=640,
    device='cuda'  # or 'cpu'
)

# Save fine-tuned model
model.save('trained/yolo_conflict_custom.pt')
```

**Dataset Requirements:**
- Labeled satellite images with bounding boxes
- Classes: weapon, military_vehicle, tank, damaged_building, etc.
- Recommended: 1000+ images minimum

### Option 2: Train LSTM Forecaster

```python
import torch
from agents.forecasting_agent import LSTMForecaster

# Prepare your data (features from Uganda refugee data)
# X: Historical features (90+ variables)
# y: Actual displacement numbers

# Initialize model
model = LSTMForecaster(
    input_size=90,  # Number of features
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

# Train (example)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

for epoch in range(100):
    # Training loop
    pass

# Save trained model
torch.save(model.state_dict(), 'trained/lstm_forecaster.pth')
```

## Model Performance

### Current (Demo with Pretrained Models):
- **YOLO Detection**: Pretrained on COCO (not conflict-specific)
- **LSTM Forecast**: Mock predictions (architecture ready)

### Expected (After Training on Real Data):
- **YOLO Precision**: 85-92% on conflict imagery (based on similar studies)
- **LSTM RMSE**: 300-500 people (4-month ahead forecast)
- **System Latency**: <60 seconds (satellite image → alert)

## Production Deployment

For production deployment, ensure:
1. ✅ YOLO model fine-tuned on conflict imagery
2. ✅ LSTM model trained on historical UNHCR data (2014-2023)
3. ✅ Feature scaler saved and versioned
4. ✅ Models uploaded to Google Cloud Storage
5. ✅ Version control for model updates

## Model Versioning

Use semantic versioning for models:
- `yolo_v1.0.0.pt` - Initial production model
- `yolo_v1.1.0.pt` - Minor improvements
- `lstm_v2.0.0.pth` - Major architecture change

## Size Considerations

Current sizes:
- YOLO (yolov8n.pt): 6.3 MB ✅
- LSTM (when trained): ~5-10 MB
- Total: ~15 MB (well within deployment limits)

For Google Cloud Run deployment:
- Container limit: 10 GB
- Model storage: Use Cloud Storage for large models
- Load models on container startup

## License

- **YOLOv8**: AGPL-3.0 (Ultralytics)
- **LSTM**: MIT (custom implementation)

---

**Note**: For the competition demo, the pretrained YOLO model is sufficient to demonstrate the multi-agent system. Training custom models would improve accuracy but is not required for submission.
