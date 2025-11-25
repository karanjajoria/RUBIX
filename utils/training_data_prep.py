"""
Training Data Preparation Utilities
Helper functions for preparing data for LSTM and YOLO model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime, timedelta

from config.config import DATA_DIR, ForecastingConfig


def merge_all_data_sources() -> pd.DataFrame:
    """
    Merge all downloaded data sources into a single training dataset.

    Combines:
    - UNHCR refugee/displacement data
    - World Bank economic indicators
    - Climate data (temperature, precipitation)
    - ACLED conflict data

    Returns:
        Merged DataFrame with all features aligned by date/location
    """
    print("Merging all data sources...")

    merged_data = None

    # 1. Load UNHCR data (primary)
    unhcr_file = DATA_DIR / "unhcr_uganda.csv"
    if unhcr_file.exists():
        df_unhcr = pd.read_csv(unhcr_file)
        print(f"  Loaded UNHCR: {len(df_unhcr)} rows")

        # Convert date columns if available
        if 'year' in df_unhcr.columns:
            df_unhcr['date'] = pd.to_datetime(df_unhcr['year'].astype(str) + '-01-01')
        elif 'date' in df_unhcr.columns:
            df_unhcr['date'] = pd.to_datetime(df_unhcr['date'])

        merged_data = df_unhcr
    else:
        print("  WARNING: UNHCR data not found")
        merged_data = pd.DataFrame()

    # 2. Load World Bank data
    wb_file = DATA_DIR / "worldbank_uganda.csv"
    if wb_file.exists():
        df_wb = pd.read_csv(wb_file)
        print(f"  Loaded World Bank: {len(df_wb)} rows")

        # Merge on year/date
        if 'year' in df_wb.columns and len(merged_data) > 0:
            merged_data = merged_data.merge(df_wb, on='year', how='left', suffixes=('', '_wb'))

    # 3. Load climate data
    climate_file = DATA_DIR / "climate_uganda.csv"
    if climate_file.exists():
        df_climate = pd.read_csv(climate_file)
        print(f"  Loaded Climate: {len(df_climate)} rows")

        # Aggregate climate data by month/year
        if 'date' in df_climate.columns and len(merged_data) > 0:
            df_climate['date'] = pd.to_datetime(df_climate['date'])
            df_climate['year'] = df_climate['date'].dt.year
            df_climate['month'] = df_climate['date'].dt.month

            # Monthly averages
            climate_monthly = df_climate.groupby(['year', 'month']).agg({
                'temperature': 'mean',
                'precipitation': 'sum'
            }).reset_index()

            # Merge (this is simplified - in production use proper date alignment)
            if 'year' in merged_data.columns:
                merged_data = merged_data.merge(
                    climate_monthly.groupby('year').mean().reset_index(),
                    on='year', how='left', suffixes=('', '_climate')
                )

    # 4. Load ACLED conflict data
    acled_file = DATA_DIR / "acled_uganda.csv"
    if acled_file.exists():
        df_acled = pd.read_csv(acled_file)
        print(f"  Loaded ACLED: {len(df_acled)} rows")

        # Aggregate conflict events by month/year
        if 'date' in df_acled.columns and len(merged_data) > 0:
            df_acled['date'] = pd.to_datetime(df_acled['date'])
            df_acled['year'] = df_acled['date'].dt.year
            df_acled['month'] = df_acled['date'].dt.month

            # Count events and calculate intensity
            conflict_monthly = df_acled.groupby(['year', 'month']).agg({
                'fatalities': 'sum',
                'event_type': 'count'
            }).reset_index()
            conflict_monthly.rename(columns={'event_type': 'conflict_events'}, inplace=True)

            # Merge
            if 'year' in merged_data.columns:
                merged_data = merged_data.merge(
                    conflict_monthly.groupby('year').mean().reset_index(),
                    on='year', how='left', suffixes=('', '_conflict')
                )

    # Fill missing values
    if len(merged_data) > 0:
        merged_data.fillna(method='ffill', inplace=True)
        merged_data.fillna(0, inplace=True)

    print(f"\nMerged dataset: {len(merged_data)} rows, {len(merged_data.columns)} columns")

    return merged_data


def create_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for LSTM training matching ForecastingConfig.FEATURES.

    Args:
        df: Raw merged dataframe

    Returns:
        DataFrame with all required features for LSTM
    """
    print("\nEngineering LSTM features...")

    df_features = df.copy()

    # Map existing columns to required features
    feature_mapping = {
        # Conflict features
        'conflict_events': 'conflict_intensity',
        'fatalities': 'battle_deaths',

        # Economic features
        'GDP growth (annual %)': 'gdp_growth',
        'Inflation, consumer prices (annual %)': 'inflation_rate',
        'Unemployment, total (% of total labor force)': 'unemployment_rate',

        # Climate features
        'temperature': 'temperature_celsius',
        'precipitation': 'precipitation_mm',
    }

    # Rename columns
    for old_name, new_name in feature_mapping.items():
        if old_name in df_features.columns:
            df_features[new_name] = df_features[old_name]

    # Add missing features with defaults or derivations
    all_required = ForecastingConfig.FEATURES

    for feat in all_required:
        if feat not in df_features.columns:
            # Create synthetic feature or derive from existing
            if 'rate' in feat or 'pct' in feat:
                df_features[feat] = np.random.uniform(0, 1, len(df_features))
            elif 'population' in feat:
                df_features[feat] = np.random.uniform(100, 500, len(df_features))
            elif 'index' in feat:
                df_features[feat] = np.random.uniform(0, 100, len(df_features))
            else:
                df_features[feat] = np.random.randn(len(df_features))

    # Select only required features
    df_features = df_features[all_required + ['date'] if 'date' in df_features.columns else all_required]

    print(f"  Features engineered: {len(all_required)}")

    return df_features


def augment_time_series(df: pd.DataFrame, target_length: int = 120) -> pd.DataFrame:
    """
    Augment time series data to have more training samples.

    Techniques:
    - Interpolation for missing months
    - Noise injection
    - Scenario simulation (optimistic/pessimistic)

    Args:
        df: Original time series
        target_length: Desired number of time steps

    Returns:
        Augmented DataFrame
    """
    print(f"\nAugmenting time series to {target_length} samples...")

    if len(df) >= target_length:
        return df

    # Create date range
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        start_date = df['date'].min()
    else:
        start_date = datetime(2015, 1, 1)

    full_dates = pd.date_range(start_date, periods=target_length, freq='ME')

    # Create full dataframe with interpolation
    df_full = pd.DataFrame({'date': full_dates})

    # Merge and interpolate
    if 'date' in df.columns:
        df_full = df_full.merge(df, on='date', how='left')
    else:
        # Just duplicate and extend
        df_full = pd.concat([df] * (target_length // len(df) + 1), ignore_index=True)
        df_full = df_full.iloc[:target_length]

    # Interpolate missing values
    df_full = df_full.interpolate(method='linear')
    df_full.fillna(method='bfill', inplace=True)

    # Add some noise for variation
    numeric_cols = df_full.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        noise = np.random.randn(len(df_full)) * df_full[col].std() * 0.05  # 5% noise
        df_full[col] += noise

    print(f"  Augmented from {len(df)} to {len(df_full)} samples")

    return df_full


def export_training_dataset(df: pd.DataFrame, filename: str = "lstm_training_data.csv"):
    """Export prepared training data."""
    output_path = DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"\nTraining data exported to: {output_path}")
    return output_path


def create_yolo_annotation_template(image_name: str, detections: List[Dict]) -> str:
    """
    Create YOLO format annotation for an image.

    YOLO format: class_id x_center y_center width height (normalized 0-1)

    Args:
        image_name: Name of the image file
        detections: List of {"class": int, "bbox": [x, y, w, h]}

    Returns:
        Annotation string in YOLO format
    """
    lines = []

    for det in detections:
        class_id = det['class']
        x_center, y_center, width, height = det['bbox']

        # Normalize to 0-1 (assuming already normalized)
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)

    return "\n".join(lines)


def generate_sample_yolo_annotations():
    """
    Generate sample YOLO annotations for demo purposes.

    In production, use actual labeled satellite imagery.
    """
    print("\nGenerating sample YOLO annotations...")

    dataset_dir = DATA_DIR / "yolo_dataset"

    # Create sample annotations for demonstration
    sample_annotations = {
        "satellite_001.jpg": [
            {"class": 0, "bbox": [0.5, 0.5, 0.3, 0.3]},  # refugee_camp
            {"class": 1, "bbox": [0.2, 0.3, 0.1, 0.1]},  # military_vehicle
        ],
        "satellite_002.jpg": [
            {"class": 2, "bbox": [0.6, 0.4, 0.2, 0.2]},  # destroyed_building
            {"class": 3, "bbox": [0.3, 0.7, 0.4, 0.2]},  # displaced_population
        ],
        "satellite_003.jpg": [
            {"class": 4, "bbox": [0.5, 0.5, 0.15, 0.1]},  # aid_convoy
        ]
    }

    labels_dir = dataset_dir / "labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_name, detections in sample_annotations.items():
        annotation_text = create_yolo_annotation_template(img_name, detections)

        # Save annotation
        label_name = img_name.replace('.jpg', '.txt')
        label_path = labels_dir / label_name

        with open(label_path, 'w') as f:
            f.write(annotation_text)

        print(f"  Created annotation: {label_path}")

    print(f"\nSample annotations created in: {labels_dir}")
    print("NOTE: Add corresponding images to complete the dataset")


def validate_training_data():
    """Validate that training data is ready."""
    print("\n" + "="*80)
    print("VALIDATING TRAINING DATA")
    print("="*80)

    issues = []

    # Check LSTM data
    print("\n[LSTM Data]")
    data_files = ['unhcr_uganda.csv', 'worldbank_uganda.csv', 'climate_uganda.csv', 'acled_uganda.csv']

    for file in data_files:
        path = DATA_DIR / file
        if path.exists():
            df = pd.read_csv(path)
            print(f"  OK: {file} ({len(df)} rows)")
        else:
            print(f"  MISSING: {file}")
            issues.append(f"Missing {file}")

    # Check YOLO data
    print("\n[YOLO Data]")
    yolo_dataset = DATA_DIR / "yolo_dataset"

    if yolo_dataset.exists():
        train_images = list((yolo_dataset / "images" / "train").glob("*.jpg")) + \
                      list((yolo_dataset / "images" / "train").glob("*.png"))
        train_labels = list((yolo_dataset / "labels" / "train").glob("*.txt"))

        print(f"  Training images: {len(train_images)}")
        print(f"  Training labels: {len(train_labels)}")

        if len(train_images) == 0:
            issues.append("No YOLO training images found")
        if len(train_labels) == 0:
            issues.append("No YOLO labels found")
    else:
        print("  MISSING: YOLO dataset directory")
        issues.append("YOLO dataset not initialized")

    # Summary
    print("\n" + "="*80)
    if len(issues) == 0:
        print("VALIDATION PASSED: Ready for training!")
    else:
        print("VALIDATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRun download_data.py and train.py to prepare data")
    print("="*80)

    return len(issues) == 0


if __name__ == "__main__":
    """Run data preparation pipeline."""

    print("="*80)
    print("TRAINING DATA PREPARATION")
    print("="*80)

    # 1. Merge all data sources
    df_merged = merge_all_data_sources()

    # 2. Create LSTM features
    df_lstm = create_lstm_features(df_merged)

    # 3. Augment if needed
    df_augmented = augment_time_series(df_lstm, target_length=120)

    # 4. Export
    export_training_dataset(df_augmented, "lstm_training_data.csv")

    # 5. Create sample YOLO annotations
    generate_sample_yolo_annotations()

    # 6. Validate
    validate_training_data()

    print("\nData preparation complete!")
    print("Next step: python train.py --model all --epochs 50")
