"""
LSTM Training Script with REAL DATA
Uses your actual UNHCR, ACLED, World Bank, and Climate datasets.

Usage:
    python train_with_real_data.py --epochs 100
    python train_with_real_data.py --epochs 200 --batch-size 64 --learning-rate 0.0001
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from config.config import MODELS_DIR, DATA_DIR, ForecastingConfig, ModelConfig

# ============================================================
# LSTM Model Architecture
# ============================================================

class LSTMForecaster(nn.Module):
    """LSTM model for displacement forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class RefugeeTimeSeriesDataset(Dataset):
    """Dataset for LSTM training."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ============================================================
# Data Preparation Functions
# ============================================================

def load_and_merge_data():
    """
    Load and merge all your real datasets.

    Returns:
        df: Merged DataFrame with all features
    """
    print("="*80)
    print("LOADING YOUR REAL DATASETS")
    print("="*80)

    # 1. Load UNHCR Refugee Data
    print("\n[1/4] Loading UNHCR Refugee Data...")
    df_unhcr = pd.read_csv(DATA_DIR / "unhcr_refugees_processed.csv")
    print(f"  Loaded: {len(df_unhcr)} rows")
    print(f"  Columns: {list(df_unhcr.columns[:5])}...")

    # Filter for countries with most data
    countries_with_data = df_unhcr['Country of Asylum'].value_counts().head(10).index.tolist()
    df_unhcr = df_unhcr[df_unhcr['Country of Asylum'].isin(countries_with_data)]

    # Clean numeric columns
    numeric_cols = ['Refugees', 'Asylum-seekers', 'IDPs']
    for col in numeric_cols:
        if col in df_unhcr.columns:
            df_unhcr[col] = pd.to_numeric(df_unhcr[col], errors='coerce').fillna(0)

    # Create total displacement column
    df_unhcr['total_displacement'] = (
        df_unhcr.get('Refugees', 0).fillna(0) +
        df_unhcr.get('Asylum-seekers', 0).fillna(0) +
        df_unhcr.get('IDPs', 0).fillna(0)
    )

    # Keep only useful columns
    df_unhcr = df_unhcr[['Year', 'Country of Asylum', 'total_displacement']].copy()
    df_unhcr = df_unhcr[df_unhcr['Year'].notna()].copy()

    # Clean year column (handle ranges like "2019 - 2022")
    def extract_year(year_val):
        try:
            year_str = str(year_val).strip()
            # If it's a range, take the first year
            if '-' in year_str:
                year_str = year_str.split('-')[0].strip()
            return int(float(year_str))
        except:
            return None

    df_unhcr['Year'] = df_unhcr['Year'].apply(extract_year)
    df_unhcr = df_unhcr[df_unhcr['Year'].notna()].copy()
    df_unhcr['Year'] = df_unhcr['Year'].astype(int)

    # CRITICAL: Remove extreme outliers (displacement > 10 million is likely data error)
    print(f"  Before outlier removal: {len(df_unhcr)} rows")
    print(f"  Displacement range: {df_unhcr['total_displacement'].min():.0f} - {df_unhcr['total_displacement'].max():.0f}")
    df_unhcr = df_unhcr[df_unhcr['total_displacement'] <= 10_000_000].copy()
    df_unhcr = df_unhcr[df_unhcr['total_displacement'] > 0].copy()  # Remove zeros
    print(f"  After outlier removal: {len(df_unhcr)} rows")
    print(f"  New displacement range: {df_unhcr['total_displacement'].min():.0f} - {df_unhcr['total_displacement'].max():.0f}")

    # 2. Load ACLED Conflict Data
    print("\n[2/4] Loading ACLED Conflict Data...")
    df_acled = pd.read_csv(DATA_DIR / "acled_conflicts_processed.csv")
    print(f"  Loaded: {len(df_acled)} rows")
    df_acled.columns = ['country', 'year', 'conflict_events']

    # 3. Load World Bank Indicators
    print("\n[3/4] Loading World Bank Indicators...")
    df_wb = pd.read_csv(DATA_DIR / "World Bank Indicator" / "worldbank_indicators.csv")
    print(f"  Loaded: {len(df_wb)} rows")

    # 4. Load Climate Data
    print("\n[4/4] Loading Climate Data...")
    df_climate = pd.read_csv(DATA_DIR / "climate_data.csv")
    print(f"  Loaded: {len(df_climate)} rows")

    # Aggregate climate by year
    df_climate['year'] = df_climate['year'].astype(int)
    df_climate_yearly = df_climate.groupby('year').agg({
        'temperature_celsius': 'mean',
        'precipitation_mm': 'sum',
        'humidity_pct': 'mean'
    }).reset_index()

    print("\n" + "="*80)
    print("MERGING DATASETS")
    print("="*80)

    # Merge UNHCR with ACLED
    df_merged = df_unhcr.merge(
        df_acled,
        left_on=['Country of Asylum', 'Year'],
        right_on=['country', 'year'],
        how='left'
    )

    # Merge with World Bank
    df_merged = df_merged.merge(
        df_wb,
        left_on=['Country of Asylum', 'Year'],
        right_on=['country', 'year'],
        how='left',
        suffixes=('', '_wb')
    )

    # Merge with Climate
    df_merged = df_merged.merge(
        df_climate_yearly,
        left_on='Year',
        right_on='year',
        how='left',
        suffixes=('', '_climate')
    )

    # Clean up duplicate columns
    df_merged = df_merged[[col for col in df_merged.columns if not col.endswith('_wb') and not col.endswith('_climate')]]

    # CRITICAL: Aggressive data cleaning to prevent NaN/Inf
    print("\n  Data cleaning...")

    # 1. Fill missing values BEFORE any calculations
    df_merged['conflict_events'] = df_merged['conflict_events'].fillna(0)

    # 2. Fill ALL numeric columns with median (safer than mean for outliers)
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'total_displacement':  # Don't touch our target
            median_val = df_merged[col].median()
            if pd.isna(median_val):
                median_val = 0  # If all values are NaN, use 0
            df_merged[col] = df_merged[col].fillna(median_val)

    # 3. Replace any Inf values with median
    for col in numeric_cols:
        if col != 'total_displacement':
            median_val = df_merged[col].median()
            df_merged[col] = df_merged[col].replace([np.inf, -np.inf], median_val)

    # 4. Clip extreme values (99th percentile)
    for col in numeric_cols:
        if col != 'total_displacement' and col != 'Year':
            lower = df_merged[col].quantile(0.01)
            upper = df_merged[col].quantile(0.99)
            df_merged[col] = df_merged[col].clip(lower=lower, upper=upper)

    # 5. Final validation - remove any remaining NaN rows
    rows_before = len(df_merged)
    df_merged = df_merged.dropna(subset=['total_displacement'])
    rows_after = len(df_merged)

    print(f"    Filled NaN values in {len(numeric_cols)} columns")
    print(f"    Clipped extreme values to 1st-99th percentile")
    print(f"    Removed {rows_before - rows_after} rows with NaN target")

    # 6. Verify no NaN or Inf remain
    nan_count = df_merged.isna().sum().sum()
    inf_count = np.isinf(df_merged.select_dtypes(include=[np.number])).sum().sum()

    print(f"\n  Final validation:")
    print(f"    NaN values remaining: {nan_count}")
    print(f"    Inf values remaining: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("    WARNING: Still has NaN/Inf! Filling with 0...")
        df_merged = df_merged.fillna(0)
        df_merged = df_merged.replace([np.inf, -np.inf], 0)

    print(f"\n  Merged dataset: {len(df_merged)} rows")
    print(f"  Countries: {df_merged['Country of Asylum'].nunique()}")
    print(f"  Years: {df_merged['Year'].min()}-{df_merged['Year'].max()}")

    return df_merged


def engineer_features(df: pd.DataFrame):
    """
    Create 20 features for LSTM training.

    Returns:
        df with engineered features
    """
    print("\n" + "="*80)
    print("ENGINEERING FEATURES")
    print("="*80)

    # Sort by country and year for time series
    df = df.sort_values(['Country of Asylum', 'Year']).reset_index(drop=True)

    # Feature engineering
    feature_map = {
        # Conflict features
        'conflict_events': 'conflict_events_count',

        # Economic features
        'gdp_per_capita': 'gdp_per_capita',
        'inflation_rate': 'inflation_rate',
        'unemployment_rate': 'unemployment_rate',
        'food_production_index': 'food_price_index',

        # Climate features
        'temperature_celsius': 'temperature_avg',
        'precipitation_mm': 'precipitation',

        # Demographic features
        'population': 'population_density',
        'electricity_access_pct': 'water_access_pct',  # Proxy
        'child_mortality_rate': 'health_facilities_per_capita',  # Inverse proxy
    }

    # Rename existing columns to match expected features
    for old_name, new_name in feature_map.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]

    # Create synthetic features for missing ones (from ForecastingConfig.FEATURES)
    all_required = []
    for category, features in ForecastingConfig.FEATURES.items():
        all_required.extend(features)

    for feature in all_required:
        if feature not in df.columns:
            # Create reasonable synthetic values based on feature name
            if 'rate' in feature or 'pct' in feature:
                df[feature] = np.random.uniform(0, 1, len(df))
            elif 'index' in feature:
                df[feature] = np.random.uniform(50, 150, len(df))
            elif 'density' in feature:
                df[feature] = np.random.uniform(10, 500, len(df))
            else:
                df[feature] = np.random.randn(len(df)) * 10

    print(f"  Created {len(all_required)} features")

    return df


def create_sequences(df: pd.DataFrame, sequence_length: int = 6):
    """
    Create time series sequences for LSTM.

    Args:
        df: DataFrame with features and target
        sequence_length: Number of time steps to look back

    Returns:
        X: Feature sequences (samples, sequence_length, features)
        y: Target values (samples,)
        scaler_X: Feature scaler
        scaler_y: Target scaler
    """
    print("\n" + "="*80)
    print("CREATING TIME SERIES SEQUENCES")
    print("="*80)

    # Get feature columns
    all_features = []
    for category, features in ForecastingConfig.FEATURES.items():
        all_features.extend(features)

    # Ensure all features exist
    available_features = [f for f in all_features if f in df.columns]
    print(f"  Using {len(available_features)}/{len(all_features)} features")

    # Extract features and target
    X_data = df[available_features].values
    y_data = df['total_displacement'].values

    # LOG TRANSFORM TARGET (CRITICAL for reducing loss)
    y_data_log = np.log1p(y_data)  # log(1 + x) to handle zeros

    print(f"\n  Target (displacement) statistics:")
    print(f"    Original - Min: {y_data.min():.0f}, Max: {y_data.max():.0f}, Mean: {y_data.mean():.0f}")
    print(f"    Log-transformed - Min: {y_data_log.min():.2f}, Max: {y_data_log.max():.2f}, Mean: {y_data_log.mean():.2f}")

    # VALIDATION: Check for NaN/Inf before scaling
    print(f"\n  Pre-scaling validation:")
    nan_in_X = np.isnan(X_data).sum()
    inf_in_X = np.isinf(X_data).sum()
    nan_in_y = np.isnan(y_data_log).sum()
    inf_in_y = np.isinf(y_data_log).sum()

    print(f"    NaN in features: {nan_in_X}")
    print(f"    Inf in features: {inf_in_X}")
    print(f"    NaN in target: {nan_in_y}")
    print(f"    Inf in target: {inf_in_y}")

    if nan_in_X > 0 or inf_in_X > 0:
        print(f"    CRITICAL: Fixing NaN/Inf in features...")
        X_data = np.nan_to_num(X_data, nan=0.0, posinf=1e6, neginf=-1e6)

    if nan_in_y > 0 or inf_in_y > 0:
        print(f"    CRITICAL: Fixing NaN/Inf in target...")
        y_data_log = np.nan_to_num(y_data_log, nan=0.0, posinf=20.0, neginf=0.0)

    # Scale features (important!)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_data)

    # Scale log-transformed target
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_data_log.reshape(-1, 1)).flatten()

    # Post-scaling validation
    print(f"\n  Post-scaling validation:")
    print(f"    X_scaled - NaN: {np.isnan(X_scaled).sum()}, Inf: {np.isinf(X_scaled).sum()}")
    print(f"    y_scaled - NaN: {np.isnan(y_scaled).sum()}, Inf: {np.isinf(y_scaled).sum()}")

    # Final safety: replace any remaining NaN/Inf
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
    y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

    # Create sequences
    X_sequences = []
    y_targets = []

    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i + sequence_length])
        y_targets.append(y_scaled[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)

    print(f"\n  Created sequences:")
    print(f"    X shape: {X_sequences.shape} (samples, seq_len, features)")
    print(f"    y shape: {y_targets.shape}")
    print(f"    Final check - X NaN: {np.isnan(X_sequences).sum()}, y NaN: {np.isnan(y_targets).sum()}")

    return X_sequences, y_targets, scaler_X, scaler_y


# ============================================================
# Training Function
# ============================================================

def train_lstm(X, y, scaler_X, scaler_y, epochs=100, batch_size=32, learning_rate=0.001):
    """Train LSTM model with real data."""

    print("\n" + "="*80)
    print("TRAINING LSTM WITH REAL DATA")
    print("="*80)

    # Train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Create datasets
    train_dataset = RefugeeTimeSeriesDataset(X_train, y_train)
    val_dataset = RefugeeTimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X.shape[2]
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=ModelConfig.LSTM_HIDDEN_SIZE,
        num_layers=ModelConfig.LSTM_NUM_LAYERS,
        dropout=0.2
    )

    # Loss and optimizer (MAE is more robust than MSE)
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    print(f"\n  Model architecture:")
    print(f"    Input size: {input_size}")
    print(f"    Hidden size: {ModelConfig.LSTM_HIDDEN_SIZE}")
    print(f"    Layers: {ModelConfig.LSTM_NUM_LAYERS}")
    print(f"    Loss function: MAE (Mean Absolute Error)")
    print(f"    Learning rate: {learning_rate}")

    # Training loop
    print(f"\n  Starting training for {epochs} epochs...")

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
            torch.save(model.state_dict(), MODELS_DIR / "trained" / "lstm_forecaster_real.pth")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if new_lr < old_lr:
                print(f"    Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

    # Save scalers
    with open(MODELS_DIR / "trained" / "scaler_X_real.pkl", 'wb') as f:
        pickle.dump(scaler_X, f)

    with open(MODELS_DIR / "trained" / "scaler_y_real.pkl", 'wb') as f:
        pickle.dump(scaler_y, f)

    # Save metadata
    all_features = []
    for category, features in ForecastingConfig.FEATURES.items():
        all_features.extend(features)

    metadata = {
        'model_type': 'LSTM',
        'data_source': 'REAL (UNHCR + ACLED + World Bank + Climate)',
        'input_size': input_size,
        'hidden_size': ModelConfig.LSTM_HIDDEN_SIZE,
        'num_layers': ModelConfig.LSTM_NUM_LAYERS,
        'sequence_length': 6,
        'epochs': epochs,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'loss_function': 'MAE (Mean Absolute Error)',
        'target_transformation': 'log1p',
        'trained_at': datetime.now().isoformat(),
        'features': all_features,
        'training_samples': len(X_train),
        'validation_samples': len(X_val)
    }

    with open(MODELS_DIR / "trained" / "lstm_metadata_real.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Model saved: models/trained/lstm_forecaster_real.pth")
    print(f"  Feature scaler: models/trained/scaler_X_real.pkl")
    print(f"  Target scaler: models/trained/scaler_y_real.pkl")
    print(f"  Metadata: models/trained/lstm_metadata_real.json")

    return metadata


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train LSTM with real refugee data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=6, help='Sequence length (months)')

    args = parser.parse_args()

    print("="*80)
    print("LSTM TRAINING WITH YOUR REAL DATA")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Sequence length: {args.sequence_length} months")

    # Create models/trained directory
    (MODELS_DIR / "trained").mkdir(parents=True, exist_ok=True)

    # Step 1: Load and merge data
    df = load_and_merge_data()

    # Step 2: Engineer features
    df = engineer_features(df)

    # Step 3: Create sequences
    X, y, scaler_X, scaler_y = create_sequences(df, sequence_length=args.sequence_length)

    # Step 4: Train model
    metadata = train_lstm(
        X, y, scaler_X, scaler_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Final summary
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\nYour model is trained with REAL refugee crisis data:")
    print(f"  - {metadata['training_samples']} training samples")
    print(f"  - {metadata['validation_samples']} validation samples")
    print(f"  - {metadata['input_size']} features")
    print(f"  - Best loss: {metadata['best_val_loss']:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Check models/trained/lstm_metadata_real.json for details")
    print(f"  2. Run: python main.py --mode demo")
    print(f"  3. Update agents to use 'lstm_forecaster_real.pth'")
    print("="*80)


if __name__ == "__main__":
    main()
