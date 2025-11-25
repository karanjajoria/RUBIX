"""
Automated Data Download Script
Downloads UNHCR, World Bank, and Climate data for the refugee crisis model.
"""

import requests
import pandas as pd
from pathlib import Path
import time
import json

# Setup
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("AI-Powered Refugee Crisis Intelligence System - Data Downloader")
print("="*70)
print()


def download_worldbank_data():
    """Download World Bank development indicators for Uganda."""
    print("[World Bank] Downloading data...")

    indicators = {
        'SP.POP.TOTL': 'population_total',
        'NY.GDP.PCAP.CD': 'gdp_per_capita',
        'FP.CPI.TOTL.ZG': 'inflation_rate',
        'AG.PRD.FOOD.XD': 'food_production_index',
        'SL.UEM.TOTL.ZS': 'unemployment_rate',
        'SP.DYN.LE00.IN': 'life_expectancy',
        'EG.ELC.ACCS.ZS': 'electricity_access_pct'
    }

    all_data = []

    for code, name in indicators.items():
        url = f"https://api.worldbank.org/v2/country/UGA/indicator/{code}?date=2014:2023&format=json&per_page=1000"

        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if len(data) > 1 and data[1]:
                    for item in data[1]:
                        all_data.append({
                            'year': item.get('date'),
                            'indicator': name,
                            'value': item.get('value')
                        })
                    print(f"  OK {name}")
                else:
                    print(f"  FAIL {name} - No data available")
            else:
                print(f"  FAIL {name} - HTTP {response.status_code}")

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"  FAIL {name} - Error: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        # Pivot to have indicators as columns
        df_pivot = df.pivot(index='year', columns='indicator', values='value').reset_index()
        df_pivot.to_csv(DATA_DIR / "worldbank_uganda.csv", index=False)
        print(f"  SAVED: worldbank_uganda.csv ({len(df_pivot)} rows)")
    else:
        print("  FAIL No World Bank data downloaded")

    print()


def download_unhcr_data():
    """Download UNHCR refugee population data for Uganda."""
    print("[UNHCR] Downloading refugee data...")

    # UNHCR API endpoint
    url = "https://api.unhcr.org/population/v1/population/"

    params = {
        'limit': 10000,
        'dataset': 'population',
        'yearFrom': 2014,
        'yearTo': 2023,
        'coo': 'UGA',  # Country of Origin: Uganda
        'columns[]': ['refugees', 'asylum_seekers', 'idps', 'returned_refugees']
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if 'items' in data and data['items']:
                df = pd.DataFrame(data['items'])
                df.to_csv(DATA_DIR / "unhcr_uganda_population.csv", index=False)
                print(f"  OK Downloaded UNHCR data")
                print(f"  SAVED: unhcr_uganda_population.csv ({len(df)} rows)")
            else:
                print("  FAIL No data in response")
                # Save sample structure for demo
                create_sample_unhcr_data()
        else:
            print(f"  FAIL HTTP {response.status_code}")
            print("  INFO  Creating sample data for demo...")
            create_sample_unhcr_data()

    except Exception as e:
        print(f"  FAIL Error: {e}")
        print("  INFO  Creating sample data for demo...")
        create_sample_unhcr_data()

    print()


def create_sample_unhcr_data():
    """Create sample UNHCR data for demo if API fails."""
    import numpy as np

    years = range(2014, 2024)
    data = []

    for year in years:
        # Simulate increasing refugee numbers over time
        base_refugees = 800000
        refugees = base_refugees + (year - 2014) * 50000 + np.random.randint(-20000, 20000)
        asylum_seekers = int(refugees * 0.1) + np.random.randint(-5000, 5000)
        idps = int(refugees * 0.3) + np.random.randint(-10000, 10000)

        data.append({
            'year': year,
            'country_of_origin': 'Uganda',
            'refugees': refugees,
            'asylum_seekers': asylum_seekers,
            'internally_displaced': idps,
            'total_displaced': refugees + asylum_seekers + idps
        })

    df = pd.DataFrame(data)
    df.to_csv(DATA_DIR / "unhcr_uganda_population.csv", index=False)
    print(f"  OK Created sample UNHCR data")
    print(f"  SAVED: unhcr_uganda_population.csv ({len(df)} rows)")


def download_climate_data():
    """Download NASA POWER climate data for Uganda."""
    print("[Climate] Downloading NASA POWER data...")

    # Uganda coordinates (approximate center)
    lat, lon = 1.3733, 32.2903

    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

    params = {
        'parameters': 'T2M,PRECTOTCORR',  # Temperature, Precipitation
        'community': 'AG',
        'longitude': lon,
        'latitude': lat,
        'start': '2014',
        'end': '2023',
        'format': 'JSON'
    }

    try:
        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 200:
            data = response.json()

            # Extract parameters
            params_data = data.get('properties', {}).get('parameter', {})

            if params_data:
                # Convert to DataFrame
                climate_records = []

                for param, values in params_data.items():
                    for date, value in values.items():
                        if value != -999:  # NASA POWER uses -999 for missing data
                            year, month = date[:4], date[4:6]
                            climate_records.append({
                                'year': int(year),
                                'month': int(month),
                                'date': f"{year}-{month}",
                                'parameter': param,
                                'value': value
                            })

                df = pd.DataFrame(climate_records)

                # Pivot to have parameters as columns
                df_pivot = df.pivot_table(
                    index=['year', 'month', 'date'],
                    columns='parameter',
                    values='value'
                ).reset_index()

                # Rename columns
                df_pivot.rename(columns={
                    'T2M': 'temperature_avg_c',
                    'PRECTOTCORR': 'precipitation_mm'
                }, inplace=True)

                df_pivot.to_csv(DATA_DIR / "climate_uganda.csv", index=False)
                print(f"  OK Downloaded climate data")
                print(f"  SAVED: climate_uganda.csv ({len(df_pivot)} rows)")
            else:
                print("  FAIL No climate data in response")
                create_sample_climate_data()
        else:
            print(f"  FAIL HTTP {response.status_code}")
            create_sample_climate_data()

    except Exception as e:
        print(f"  FAIL Error: {e}")
        print("  INFO  Creating sample climate data...")
        create_sample_climate_data()

    print()


def create_sample_climate_data():
    """Create sample climate data if API fails."""
    import numpy as np

    years = range(2014, 2024)
    months = range(1, 13)
    data = []

    for year in years:
        for month in months:
            # Simulate realistic Uganda climate
            temp = 23 + np.random.uniform(-3, 3)  # 20-26°C average
            precip = 100 + np.random.uniform(-50, 100)  # 50-200mm

            data.append({
                'year': year,
                'month': month,
                'date': f"{year}-{month:02d}",
                'temperature_avg_c': round(temp, 1),
                'precipitation_mm': round(precip, 1)
            })

    df = pd.DataFrame(data)
    df.to_csv(DATA_DIR / "climate_uganda.csv", index=False)
    print(f"  OK Created sample climate data")
    print(f"  SAVED: climate_uganda.csv ({len(df)} rows)")


def create_sample_conflict_data():
    """Create sample conflict data (ACLED alternative)."""
    print("[ACLED] Creating sample conflict data...")

    import numpy as np

    years = range(2014, 2024)
    months = range(1, 13)
    data = []

    for year in years:
        for month in months:
            # Simulate conflict events
            num_events = np.random.randint(5, 30)

            for _ in range(num_events):
                event_types = ['Battles', 'Violence against civilians', 'Protests', 'Riots', 'Strategic developments']
                event_type = np.random.choice(event_types, p=[0.2, 0.3, 0.2, 0.15, 0.15])

                fatalities = 0
                if event_type in ['Battles', 'Violence against civilians']:
                    fatalities = np.random.randint(0, 20)

                data.append({
                    'year': year,
                    'month': month,
                    'date': f"{year}-{month:02d}",
                    'event_type': event_type,
                    'fatalities': fatalities,
                    'location': 'Northern Uganda',
                    'latitude': 2.0 + np.random.uniform(-1, 1),
                    'longitude': 32.0 + np.random.uniform(-1, 1)
                })

    df = pd.DataFrame(data)
    df.to_csv(DATA_DIR / "acled_uganda_conflicts.csv", index=False)
    print(f"  OK Created sample conflict data")
    print(f"  SAVED: acled_uganda_conflicts.csv ({len(df)} rows)")
    print()


def download_all():
    """Download all datasets."""
    print("Starting automated data download...")
    print(f"Data will be saved to: {DATA_DIR.absolute()}")
    print()

    # Download each dataset
    download_worldbank_data()
    download_unhcr_data()
    download_climate_data()
    create_sample_conflict_data()  # ACLED requires registration, so use sample

    print("="*70)
    print("DONE Data download complete!")
    print("="*70)
    print()
    print("Downloaded files:")
    for file in sorted(DATA_DIR.glob("*.csv")):
        size_kb = file.stat().st_size / 1024
        print(f"  FILE: {file.name} ({size_kb:.1f} KB)")

    print()
    print("Next steps:")
    print("  1. Review the data in: data/raw/")
    print("  2. Run: python main.py --mode demo")
    print("  3. For production, merge data with: python process_data.py")


if __name__ == "__main__":
    download_all()
