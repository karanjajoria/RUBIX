"""
Process Your Downloaded Data
Converts and prepares your UNHCR and ACLED data for training.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Paths
DATA_DIR = Path("data")

print("="*80)
print("PROCESSING YOUR DOWNLOADED DATA")
print("="*80)

# ============================================================
# 1. Process UNHCR Refugee Data
# ============================================================

print("\n[1/3] PROCESSING UNHCR REFUGEE DATA")
print("-"*80)

unhcr_dir = DATA_DIR / "UNHCR Refugee Data"
if unhcr_dir.exists():
    # Find all CSV/Excel files
    csv_files = list(unhcr_dir.glob("*.csv"))
    excel_files = list(unhcr_dir.glob("*.xlsx")) + list(unhcr_dir.glob("*.xls"))

    all_unhcr_data = []

    # Process CSV files
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
            print(f"  [OK] {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
            all_unhcr_data.append(df)
        except Exception as e:
            print(f"  [ERROR] {csv_file.name}: {e}")

    # Process Excel files
    for excel_file in excel_files:
        try:
            df = pd.read_excel(excel_file)
            print(f"  [OK] {excel_file.name}: {len(df)} rows, {len(df.columns)} columns")
            all_unhcr_data.append(df)
        except Exception as e:
            print(f"  [ERROR] {excel_file.name}: {e}")

    if all_unhcr_data:
        # Combine all UNHCR data
        df_unhcr_combined = pd.concat(all_unhcr_data, ignore_index=True)

        # Save processed data
        output_path = DATA_DIR / "unhcr_refugees_processed.csv"
        df_unhcr_combined.to_csv(output_path, index=False)

        print(f"\n  SAVED: {output_path}")
        print(f"  Total rows: {len(df_unhcr_combined)}")
        print(f"  Columns: {list(df_unhcr_combined.columns[:10])}...")  # Show first 10 columns
    else:
        print("  [WARNING] No UNHCR data files found")
else:
    print(f"  [ERROR] Directory not found: {unhcr_dir}")

# ============================================================
# 2. Process ACLED Conflict Data
# ============================================================

print("\n[2/3] PROCESSING ACLED CONFLICT DATA")
print("-"*80)

acled_dir = DATA_DIR / "ACLED Conflict Events"
acled_file = acled_dir / "ACLED Conflict Events Number of political violence events by country-year.xlsx"

if acled_file.exists():
    try:
        # Read Excel file
        df_acled = pd.read_excel(acled_file)
        print(f"  [OK] Loaded: {acled_file.name}")
        print(f"  Rows: {len(df_acled)}, Columns: {len(df_acled.columns)}")

        # Display first few columns to understand structure
        print(f"\n  Columns: {list(df_acled.columns)}")

        # Save as CSV for easier processing
        output_path = DATA_DIR / "acled_conflicts_processed.csv"
        df_acled.to_csv(output_path, index=False)

        print(f"\n  SAVED: {output_path}")

        # Show sample data
        print(f"\n  Sample data (first 5 rows):")
        print(df_acled.head())

    except Exception as e:
        print(f"  [ERROR] Cannot read ACLED file: {e}")
else:
    print(f"  [ERROR] File not found: {acled_file}")
    print(f"  Looking for alternative files...")

    if acled_dir.exists():
        all_files = list(acled_dir.glob("*.xlsx")) + list(acled_dir.glob("*.xls")) + list(acled_dir.glob("*.csv"))
        if all_files:
            print(f"  Found files:")
            for f in all_files:
                print(f"    - {f.name}")

# ============================================================
# 3. Create Climate Data (Synthetic for now)
# ============================================================

print("\n[3/3] CREATING CLIMATE DATA")
print("-"*80)

# Since NASA POWER API failed, create realistic synthetic climate data
# based on East Africa climate patterns

print("  Creating synthetic climate data for East Africa...")

months = pd.date_range('2014-01-01', '2023-12-31', freq='ME')

climate_data = []

locations = {
    'Uganda_Central': {'lat': 0.3476, 'lon': 32.5825, 'base_temp': 25, 'base_precip': 100},
    'Uganda_North': {'lat': 3.1864, 'lon': 32.3144, 'base_temp': 27, 'base_precip': 80},
    'South_Sudan': {'lat': 4.8594, 'lon': 31.5713, 'base_temp': 30, 'base_precip': 60},
    'DRC_East': {'lat': -1.6872, 'lon': 29.2208, 'base_temp': 23, 'base_precip': 120},
    'Kenya_North': {'lat': 3.1190, 'lon': 35.6089, 'base_temp': 28, 'base_precip': 50}
}

for i, date in enumerate(months):
    for location_name, loc_info in locations.items():
        # Seasonal variations
        month = date.month
        seasonal_temp = 3 * np.sin((month - 4) * 2 * np.pi / 12)  # Peak in April
        seasonal_precip_factor = 1 + 0.5 * np.sin((month - 4) * 2 * np.pi / 6)  # Bimodal rain

        # Random variations
        temp_noise = np.random.randn() * 2
        precip_noise = np.random.rand() * 30

        # Add drought events (some months with very low rainfall)
        if np.random.rand() < 0.1:  # 10% chance of drought month
            precip_multiplier = 0.2
        else:
            precip_multiplier = 1.0

        temperature = loc_info['base_temp'] + seasonal_temp + temp_noise
        precipitation = (loc_info['base_precip'] * seasonal_precip_factor + precip_noise) * precip_multiplier
        humidity = 60 + 20 * (precipitation / 100) + np.random.randn() * 5

        climate_data.append({
            'location': location_name,
            'latitude': loc_info['lat'],
            'longitude': loc_info['lon'],
            'year': date.year,
            'month': date.month,
            'date': date.strftime('%Y-%m-%d'),
            'temperature_celsius': round(temperature, 2),
            'precipitation_mm': round(max(0, precipitation), 2),
            'humidity_pct': round(np.clip(humidity, 0, 100), 2)
        })

df_climate = pd.DataFrame(climate_data)

output_path = DATA_DIR / "climate_data.csv"
df_climate.to_csv(output_path, index=False)

print(f"  [OK] Created climate data")
print(f"  SAVED: {output_path}")
print(f"  Rows: {len(df_climate)}")
print(f"  Locations: {df_climate['location'].nunique()}")
print(f"  Date range: {df_climate['date'].min()} to {df_climate['date'].max()}")

# ============================================================
# 4. Final Summary
# ============================================================

print("\n" + "="*80)
print("PROCESSING COMPLETE - DATA SUMMARY")
print("="*80)

datasets = {
    'UNHCR Refugees': DATA_DIR / "unhcr_refugees_processed.csv",
    'ACLED Conflicts': DATA_DIR / "acled_conflicts_processed.csv",
    'World Bank Indicators': DATA_DIR / "worldbank_indicators.csv",
    'Climate Data': DATA_DIR / "climate_data.csv"
}

all_ready = True

for name, path in datasets.items():
    if path.exists():
        try:
            df = pd.read_csv(path)
            print(f"\n[OK] {name}")
            print(f"     File: {path.name}")
            print(f"     Rows: {len(df):,}")
            print(f"     Columns: {len(df.columns)}")
        except:
            print(f"\n[ERROR] {name} - Cannot read")
            all_ready = False
    else:
        print(f"\n[MISSING] {name}")
        all_ready = False

print("\n" + "="*80)
if all_ready:
    print("ALL DATASETS READY FOR TRAINING!")
    print("\nNext steps:")
    print("  1. python train.py --model lstm --epochs 100")
    print("  2. python main.py --mode demo")
else:
    print("Some datasets are missing. Check errors above.")
print("="*80)
