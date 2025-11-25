"""
Model Training Script
Trains YOLO and LSTM models for the Refugee Crisis Intelligence System.

Usage:
    python train.py --model yolo --epochs 50
    python train.py --model lstm --epochs 100
    python train.py --model all --epochs 50
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

from config.config import MODELS_DIR, DATA_DIR, ForecastingConfig, ModelConfig, GEMINI_API_KEY

# Only import YOLO if ultralytics is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO training will be skipped.")


# ============================================================
# LSTM Model Architecture (same as in forecasting_agent.py)
# ============================================================

class LSTMForecaster(nn.Module):
    """LSTM model for displacement forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Take the last time step
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# ============================================================
# Dataset Classes
# ============================================================

class RefugeeTimeSeriesDataset(Dataset):
    """Dataset for LSTM training with time series refugee data."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ============================================================
# LSTM Training Functions
# ============================================================

def prepare_lstm_data(data_dir: Path, sequence_length: int = 6) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare time series data for LSTM training.

    Uses the downloaded data from download_data.py:
    - UNHCR refugee data
    - World Bank indicators
    - Climate data
    - ACLED conflict data

    Returns:
        X: Input sequences (samples, sequence_length, features)
        y: Target values (samples,)
        scaler: Fitted StandardScaler for features
    """
    print("\n" + "="*80)
    print("PREPARING LSTM TRAINING DATA")
    print("="*80)

    # Load all available data
    unhcr_file = data_dir / "unhcr_uganda.csv"
    worldbank_file = data_dir / "worldbank_uganda.csv"
    climate_file = data_dir / "climate_uganda.csv"
    acled_file = data_dir / "acled_uganda.csv"

    # Start with UNHCR data (our target variable)
    if not unhcr_file.exists():
        print(f"UNHCR data not found at {unhcr_file}")
        print("Generating synthetic training data for demo purposes...")
        df_train = create_synthetic_training_data(n_months=120)
        # Skip to feature creation
        # Flatten the feature dict
        feature_cols = []
        for category, features in ForecastingConfig.FEATURES.items():
            feature_cols.extend(features)

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_train[feature_cols])

        # Target: displacement
        targets = df_train['displacement'].values

        # Create sequences
        X, y = create_sequences(features_scaled, targets, sequence_length)

        print(f"\nPrepared synthetic data:")
        print(f"  Input sequences: {X.shape}")
        print(f"  Targets: {y.shape}")
        print(f"  Features: {len(feature_cols)}")

        return X, y, scaler

    df_unhcr = pd.read_csv(unhcr_file)
    print(f"\nLoaded UNHCR data: {len(df_unhcr)} rows")

    # Create synthetic time series if real data is limited
    # (In production, you'd use real historical data)
    if len(df_unhcr) < 50:
        print("Creating synthetic training data (real data is limited)...")
        df_train = create_synthetic_training_data(n_months=120)
    else:
        df_train = prepare_real_data(df_unhcr, worldbank_file, climate_file, acled_file)

    print(f"Total training samples: {len(df_train)} months")

    # Extract features matching ForecastingConfig.FEATURES
    feature_cols = [col for col in df_train.columns if col != 'displacement' and col != 'date']

    # Ensure we have all required features
    required_features = ForecastingConfig.FEATURES
    for feat in required_features:
        if feat not in df_train.columns:
            df_train[feat] = np.random.randn(len(df_train)) * 0.1  # Small random noise

    feature_cols = required_features

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_train[feature_cols])

    # Target: displacement (next month)
    if 'displacement' in df_train.columns:
        targets = df_train['displacement'].values
    else:
        # Use first feature as proxy if displacement not available
        targets = df_train[feature_cols[0]].values

    # Create sequences
    X, y = create_sequences(features_scaled, targets, sequence_length)

    print(f"\nPrepared data:")
    print(f"  Input sequences: {X.shape} (samples, sequence_length, features)")
    print(f"  Targets: {y.shape}")
    print(f"  Features: {len(feature_cols)}")

    return X, y, scaler


def create_synthetic_training_data(n_months: int = 120) -> pd.DataFrame:
    """Create synthetic refugee displacement data for training."""

    dates = pd.date_range('2015-01-01', periods=n_months, freq='ME')

    # Create realistic patterns
    base_displacement = 50000
    trend = np.linspace(0, 30000, n_months)  # Increasing trend
    seasonal = 10000 * np.sin(np.arange(n_months) * 2 * np.pi / 12)  # Yearly seasonality
    conflict_spikes = np.random.poisson(5, n_months) * 5000  # Random conflict events
    noise = np.random.randn(n_months) * 2000

    displacement = base_displacement + trend + seasonal + conflict_spikes + noise
    displacement = np.maximum(displacement, 0)  # No negative displacement

    # Create corresponding features
    data = {
        'date': dates,
        'displacement': displacement,
        'conflict_intensity': np.random.rand(n_months) * 10,
        'food_insecurity_rate': 0.3 + np.random.rand(n_months) * 0.4,
        'gdp_growth': np.random.randn(n_months) * 2 + 3,
        'unemployment_rate': 0.15 + np.random.rand(n_months) * 0.15,
        'inflation_rate': np.random.rand(n_months) * 10,
        'precipitation_mm': 50 + np.random.rand(n_months) * 100,
        'temperature_celsius': 25 + np.random.randn(n_months) * 5,
        'population_density': 150 + np.arange(n_months) * 0.5,
        'urban_population_pct': 0.20 + np.arange(n_months) * 0.001,
        'school_enrollment_rate': 0.70 + np.random.rand(n_months) * 0.1,
    }

    # Add all required features from ForecastingConfig
    # Flatten the nested dict of features
    all_features = []
    for category, features in ForecastingConfig.FEATURES.items():
        all_features.extend(features)

    for feat in all_features:
        if feat not in data:
            data[feat] = np.random.randn(n_months) * 0.5

    return pd.DataFrame(data)


def prepare_real_data(df_unhcr: pd.DataFrame, wb_file: Path,
                     climate_file: Path, acled_file: Path) -> pd.DataFrame:
    """Merge real data sources into training dataset."""

    # In production, you'd merge all sources by date/location
    # For now, use UNHCR as base and add synthetic features
    df = df_unhcr.copy()

    # Add synthetic features (replace with real merges in production)
    for feat in ForecastingConfig.FEATURES:
        if feat not in df.columns:
            df[feat] = np.random.randn(len(df)) * 0.5

    return df


def create_sequences(features: np.ndarray, targets: np.ndarray,
                    sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for LSTM."""

    X, y = [], []

    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length])

    return np.array(X), np.array(y)


def train_lstm_model(epochs: int = 100, batch_size: int = 32,
                    learning_rate: float = 0.001) -> Dict[str, Any]:
    """
    Train LSTM model for displacement forecasting.

    Returns:
        Training results and metrics
    """
    print("\n" + "="*80)
    print("TRAINING LSTM FORECASTER")
    print("="*80)

    # Prepare data
    X, y, scaler = prepare_lstm_data(DATA_DIR, sequence_length=6)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Create datasets
    train_dataset = RefugeeTimeSeriesDataset(X_train, y_train)
    val_dataset = RefugeeTimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X.shape[2]  # Number of features
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=ModelConfig.LSTM_HIDDEN_SIZE,
        num_layers=ModelConfig.LSTM_NUM_LAYERS,
        dropout=0.2
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Model architecture: {ModelConfig.LSTM_NUM_LAYERS} layers, {ModelConfig.LSTM_HIDDEN_SIZE} hidden units")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "trained" / "lstm_forecaster.pth")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if new_lr < old_lr:
                print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

    # Save scaler
    with open(MODELS_DIR / "trained" / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    # Save training metadata
    # Flatten features for metadata
    all_features = []
    for category, features in ForecastingConfig.FEATURES.items():
        all_features.extend(features)

    metadata = {
        'model_type': 'LSTM',
        'input_size': input_size,
        'hidden_size': ModelConfig.LSTM_HIDDEN_SIZE,
        'num_layers': ModelConfig.LSTM_NUM_LAYERS,
        'sequence_length': 6,
        'epochs': epochs,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'trained_at': datetime.now().isoformat(),
        'features': all_features
    }

    with open(MODELS_DIR / "trained" / "lstm_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("LSTM TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODELS_DIR / 'trained' / 'lstm_forecaster.pth'}")
    print(f"Scaler saved to: {MODELS_DIR / 'trained' / 'scaler.pkl'}")
    print(f"Metadata saved to: {MODELS_DIR / 'trained' / 'lstm_metadata.json'}")

    return metadata


# ============================================================
# YOLO Training Functions
# ============================================================

def prepare_yolo_dataset() -> Path:
    """
    Prepare YOLO training dataset for conflict detection.

    In production, you'd need:
    - Satellite images labeled with bounding boxes
    - YOLO format annotations (class x_center y_center width height)
    - data.yaml configuration file

    Returns:
        Path to dataset directory
    """
    print("\n" + "="*80)
    print("PREPARING YOLO TRAINING DATA")
    print("="*80)

    dataset_dir = DATA_DIR / "yolo_dataset"
    dataset_dir.mkdir(exist_ok=True)

    # Create directory structure
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Create data.yaml
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'refugee_camp',
            1: 'military_vehicle',
            2: 'destroyed_building',
            3: 'displaced_population',
            4: 'aid_convoy'
        }
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write("names:\n")
        for k, v in data_yaml['names'].items():
            f.write(f"  {k}: {v}\n")

    print(f"\nYOLO dataset directory created: {dataset_dir}")
    print(f"data.yaml saved: {yaml_path}")
    print(f"\nNOTE: You need to add labeled satellite images to:")
    print(f"  - {dataset_dir / 'images' / 'train'}/")
    print(f"  - {dataset_dir / 'images' / 'val'}/")
    print(f"  - {dataset_dir / 'labels' / 'train'}/  (YOLO format annotations)")
    print(f"  - {dataset_dir / 'labels' / 'val'}/  (YOLO format annotations)")
    print(f"\nFor demo purposes, this creates the structure. In production:")
    print(f"  1. Collect satellite imagery from crisis regions")
    print(f"  2. Label with tools like LabelImg or Roboflow")
    print(f"  3. Export in YOLO format")

    return dataset_dir


def train_yolo_model(epochs: int = 50, img_size: int = 640) -> Dict[str, Any]:
    """
    Train YOLO model for conflict/refugee camp detection.

    Returns:
        Training results and metrics
    """
    if not YOLO_AVAILABLE:
        print("\nERROR: ultralytics not installed. Install with: pip install ultralytics")
        return {"error": "ultralytics not available"}

    print("\n" + "="*80)
    print("TRAINING YOLO CONFLICT DETECTOR")
    print("="*80)

    # Prepare dataset
    dataset_dir = prepare_yolo_dataset()
    yaml_path = dataset_dir / "data.yaml"

    # Check if dataset has images
    train_images = list((dataset_dir / "images" / "train").glob("*.jpg")) + \
                   list((dataset_dir / "images" / "train").glob("*.png"))

    if len(train_images) == 0:
        print("\n" + "="*80)
        print("WARNING: No training images found!")
        print("="*80)
        print("YOLO training requires labeled satellite images.")
        print("Skipping YOLO training. Using pretrained YOLOv8n instead.")
        print(f"\nTo train YOLO in production:")
        print(f"  1. Add labeled images to: {dataset_dir / 'images' / 'train'}/")
        print(f"  2. Add annotations to: {dataset_dir / 'labels' / 'train'}/")
        print(f"  3. Run: python train.py --model yolo --epochs 50")

        return {
            "status": "skipped",
            "reason": "no_training_data",
            "message": "Using pretrained YOLOv8n model"
        }

    # Load pretrained YOLO model
    print(f"\nLoading pretrained YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    # Train
    print(f"\nStarting YOLO training for {epochs} epochs...")
    print(f"Dataset: {yaml_path}")
    print(f"Image size: {img_size}x{img_size}")

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        name='refugee_crisis_detector',
        project=str(MODELS_DIR / "trained"),
        patience=20,
        save=True,
        device='cpu',  # Change to 'cuda' if GPU available
        verbose=True
    )

    # Save best model to our models directory
    best_model_path = MODELS_DIR / "trained" / "refugee_crisis_detector" / "weights" / "best.pt"
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, MODELS_DIR / "trained" / "yolo_conflict_custom.pt")
        print(f"\nBest model copied to: {MODELS_DIR / 'trained' / 'yolo_conflict_custom.pt'}")

    # Save training metadata
    metadata = {
        'model_type': 'YOLOv8n',
        'task': 'object_detection',
        'classes': ['refugee_camp', 'military_vehicle', 'destroyed_building',
                   'displaced_population', 'aid_convoy'],
        'epochs': epochs,
        'img_size': img_size,
        'trained_at': datetime.now().isoformat(),
    }

    with open(MODELS_DIR / "trained" / "yolo_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("YOLO TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model saved to: {MODELS_DIR / 'trained' / 'yolo_conflict_custom.pt'}")
    print(f"Metadata saved to: {MODELS_DIR / 'trained' / 'yolo_metadata.json'}")

    return metadata


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train models for Refugee Crisis Intelligence System')
    parser.add_argument('--model', type=str, choices=['lstm', 'yolo', 'all'],
                       default='all', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for LSTM training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for LSTM')

    args = parser.parse_args()

    # Create models/trained directory
    (MODELS_DIR / "trained").mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("REFUGEE CRISIS INTELLIGENCE SYSTEM - MODEL TRAINING")
    print("="*80)
    print(f"\nTraining configuration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")

    results = {}

    # Train LSTM
    if args.model in ['lstm', 'all']:
        try:
            lstm_results = train_lstm_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            results['lstm'] = lstm_results
        except Exception as e:
            print(f"\nERROR training LSTM: {e}")
            results['lstm'] = {"error": str(e)}

    # Train YOLO
    if args.model in ['yolo', 'all']:
        try:
            yolo_results = train_yolo_model(epochs=args.epochs)
            results['yolo'] = yolo_results
        except Exception as e:
            print(f"\nERROR training YOLO: {e}")
            results['yolo'] = {"error": str(e)}

    # Save overall results
    results_file = MODELS_DIR / "trained" / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_file}")
    print(f"\nNext steps:")
    print(f"  1. Review training metrics in: {MODELS_DIR / 'trained'}/")
    print(f"  2. Test models with: python main.py --mode demo")
    print(f"  3. Agents will automatically use trained models if available")


if __name__ == "__main__":
    main()
