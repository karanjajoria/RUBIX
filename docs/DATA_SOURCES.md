# Data Sources Guide
## AI-Powered Refugee Crisis Intelligence System

This guide provides all the data sources you need for the project.

---

## 📊 Essential Datasets (For Demo & Production)

### 1. UNHCR Refugee Population Data ✅ (You found this!)

**Your Link:** Good for displacement statistics!

**Recommended Download:**
- URL: https://www.unhcr.org/refugee-statistics/download/?url=2Z8Do6
- **What to download:** CSV format, Population statistics
- **Filters to use:**
  - Years: 2014-2023 (10 years of data)
  - Countries of Origin: Uganda, South Sudan, DRC, Syria (conflict zones)
  - Population Type: All types (REF, IDP, ASY)
  - Display Type: Time series by year

**Direct Download Links:**
1. **Main Statistics Database:**
   ```
   https://api.unhcr.org/population/v1/population/?limit=10000&dataset=population&displayType=totals&columns[]=refugees&columns[]=asylum_seekers&columns[]=idps&yearFrom=2014&yearTo=2023&coo=UGA
   ```
   - Format: JSON (convert to CSV with script below)

2. **Alternative CSV Download (easier):**
   - Go to: https://data.unhcr.org/en/situations
   - Select "Uganda Situation" or "East Africa"
   - Download "Refugee Population Statistics" CSV

---

### 2. ACLED Conflict Data (Armed Conflict Location & Event Data)

**Why needed:** Provides conflict events that trigger displacement

**Download:**
1. Go to: https://acleddata.com/data-export-tool/
2. **Free registration required** (academic/research use)
3. **Filters:**
   - Region: East Africa (Uganda, South Sudan, DRC, Kenya)
   - Countries: Uganda, South Sudan, DRC
   - Event Date: 2014-01-01 to 2023-12-31
   - Event Types: All (Battles, Violence against civilians, Strategic developments)
   - Format: CSV

**Direct API Access (after registration):**
```
https://api.acleddata.com/acled/read?key=YOUR_API_KEY&email=YOUR_EMAIL&iso=UGA&year=2014|2015|2016|2017|2018|2019|2020|2021|2022|2023
```

**Alternative (if ACLED registration is slow):**
- Uppsala Conflict Data Program: https://ucdp.uu.se/downloads/
- Download: "UCDP Georeferenced Event Dataset" (CSV)

---

### 3. World Bank Socio-Economic Indicators

**Why needed:** 90+ variables for forecasting (GDP, food prices, population)

**Download:**
1. **World Development Indicators API:**
   ```
   https://api.worldbank.org/v2/country/UGA/indicator/SP.POP.TOTL;NY.GDP.PCAP.CD;FP.CPI.TOTL.ZG?date=2014:2023&format=csv
   ```

2. **Bulk Download (recommended):**
   - Go to: https://databank.worldbank.org/source/world-development-indicators
   - Select Countries: Uganda, South Sudan, DRC, Kenya
   - Select Indicators:
     - Population, total (SP.POP.TOTL)
     - GDP per capita (NY.GDP.PCAP.CD)
     - Inflation, consumer prices (FP.CPI.TOTL.ZG)
     - Food production index (AG.PRD.FOOD.XD)
     - Access to electricity (EG.ELC.ACCS.ZS)
     - Mortality rate, under-5 (SH.DYN.MORT)
     - Unemployment, total (SL.UEM.TOTL.ZS)
   - Time: 2014-2023
   - Format: CSV

3. **Quick Download (Pre-selected indicators):**
   - Download link: https://databank.worldbank.org/data/download/WDI_csv.zip
   - Extract and filter for Uganda/East Africa

---

### 4. Climate & Environmental Data

**Why needed:** Drought, temperature, precipitation affect displacement

**Download:**

1. **NOAA Climate Data:**
   - URL: https://www.ncei.noaa.gov/cdo-web/datasets
   - Dataset: "Global Summary of the Month" (GSOM)
   - Location: Uganda
   - Date Range: 2014-2023
   - Format: CSV

2. **NASA POWER Data (easier, API-based):**
   ```python
   # Use this Python script to download
   import requests
   import pandas as pd

   # Uganda coordinates
   lat, lon = 1.3733, 32.2903

   url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start=2014&end=2023&format=JSON"

   response = requests.get(url)
   data = response.json()
   # Convert to DataFrame and save as CSV
   ```

3. **Simplified Alternative:**
   - World Bank Climate Portal: https://climateknowledgeportal.worldbank.org/download-data
   - Country: Uganda
   - Download: Temperature & Precipitation (CSV)

---

### 5. Satellite Imagery (Optional for Demo)

**For Demo:** Use sample images or placeholders
**For Production:** Need real satellite imagery

**Free Sources:**

1. **Sentinel Hub (ESA Copernicus):**
   - URL: https://apps.sentinel-hub.com/eo-browser/
   - Select: Uganda region
   - Dates: 2014-2023
   - Type: Sentinel-2 (10m resolution, RGB)
   - Download: JPEG/PNG for specific dates
   - **Free tier:** 5000 requests/month

2. **NASA Earthdata:**
   - URL: https://search.earthdata.nasa.gov/
   - Dataset: "MODIS/Terra Surface Reflectance Daily"
   - Region: East Africa
   - Format: GeoTIFF
   - **Registration required** (free)

3. **For Demo (Recommended):**
   - Use Google Earth Engine samples or
   - Download a few sample images manually
   - Store in `data/sample/satellite_*.jpg`

---

## 🚀 Quick Download Script

I'll create a Python script to automate most downloads:

```python
# Save as: download_data.py
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. World Bank Data (GDP, Population, etc.)
def download_worldbank_data():
    indicators = {
        'SP.POP.TOTL': 'population',
        'NY.GDP.PCAP.CD': 'gdp_per_capita',
        'FP.CPI.TOTL.ZG': 'inflation',
        'AG.PRD.FOOD.XD': 'food_production'
    }

    for code, name in indicators.items():
        url = f"https://api.worldbank.org/v2/country/UGA/indicator/{code}?date=2014:2023&format=json&per_page=1000"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:
                df = pd.DataFrame(data[1])
                df.to_csv(DATA_DIR / f"worldbank_{name}.csv", index=False)
                print(f"✓ Downloaded: {name}")

# 2. UNHCR Data (using public API)
def download_unhcr_data():
    url = "https://api.unhcr.org/population/v1/population/?limit=10000&dataset=population&yearFrom=2014&yearTo=2023&coo=UGA"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get('items', []))
        df.to_csv(DATA_DIR / "unhcr_uganda_population.csv", index=False)
        print("✓ Downloaded: UNHCR data")

# 3. Climate Data (NASA POWER)
def download_climate_data():
    lat, lon = 1.3733, 32.2903  # Uganda
    url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start=2014&end=2023&format=JSON"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract parameters
        params = data.get('properties', {}).get('parameter', {})

        # Convert to DataFrame
        climate_data = []
        for param, values in params.items():
            for date, value in values.items():
                climate_data.append({
                    'date': date,
                    'parameter': param,
                    'value': value
                })

        df = pd.DataFrame(climate_data)
        df = df.pivot(index='date', columns='parameter', values='value').reset_index()
        df.to_csv(DATA_DIR / "climate_uganda.csv", index=False)
        print("✓ Downloaded: Climate data")

if __name__ == "__main__":
    print("Starting data download...")
    print("-" * 50)

    try:
        download_worldbank_data()
    except Exception as e:
        print(f"✗ World Bank download failed: {e}")

    try:
        download_unhcr_data()
    except Exception as e:
        print(f"✗ UNHCR download failed: {e}")

    try:
        download_climate_data()
    except Exception as e:
        print(f"✗ Climate download failed: {e}")

    print("-" * 50)
    print("Download complete! Check the 'data/raw/' folder")
```

---

## 📥 Manual Download Checklist

If automated download fails, download manually:

### Priority 1 (Essential for Demo):
- [ ] **UNHCR Uganda Refugee Data** (2014-2023)
  - Link: https://data.unhcr.org/en/situations/uganda
  - File: Download "Population Statistics" CSV

- [ ] **World Bank Uganda Indicators** (GDP, Population)
  - Link: https://databank.worldbank.org/source/world-development-indicators
  - Select: Uganda, 2014-2023, Basic indicators

### Priority 2 (Improves Accuracy):
- [ ] **ACLED Conflict Events** (Uganda, 2014-2023)
  - Link: https://acleddata.com/data-export-tool/
  - Register (free) and download CSV

- [ ] **Climate Data** (Temperature, Precipitation)
  - Link: https://climateknowledgeportal.worldbank.org/
  - Download Uganda climate CSV

### Priority 3 (Optional for Demo):
- [ ] **Satellite Images** (3-5 sample images)
  - Link: https://apps.sentinel-hub.com/eo-browser/
  - Download a few PNG/JPEG images of Uganda

---

## 📂 Expected Data Structure

After downloading, organize like this:

```
Google-Kaggle/
└── data/
    ├── raw/                           # Raw downloaded data
    │   ├── unhcr_uganda_population.csv
    │   ├── acled_uganda_conflicts.csv
    │   ├── worldbank_gdp_per_capita.csv
    │   ├── worldbank_population.csv
    │   ├── climate_uganda.csv
    │   └── ...
    │
    ├── processed/                     # Cleaned & merged data
    │   └── uganda_features_2014_2023.csv
    │
    └── sample/                        # Sample satellite images
        ├── satellite_1.jpg
        ├── satellite_2.jpg
        └── satellite_3.jpg
```

---

## 🔄 Data Processing Pipeline

After downloading, run this script to merge all data:

```python
# Save as: process_data.py
import pandas as pd
from pathlib import Path

# Load all raw data
unhcr = pd.read_csv("data/raw/unhcr_uganda_population.csv")
worldbank = pd.read_csv("data/raw/worldbank_gdp_per_capita.csv")
climate = pd.read_csv("data/raw/climate_uganda.csv")

# Merge by year
# ... (processing logic)

# Save combined dataset
merged_df.to_csv("data/processed/uganda_features_2014_2023.csv", index=False)
```

---

## ⚡ Quick Start (Minimal Data)

**Don't have time to download everything?**

Use the demo with synthetic data (already built into `main.py`):

```bash
python main.py --mode demo
```

This uses mock data that simulates real patterns. **Perfect for testing the multi-agent system!**

---

## 🎯 For Competition Submission

**What judges expect:**

1. ✅ **Demo works with sample/synthetic data** (this is fine!)
2. ✅ **Code can process real data** (show data pipeline in code)
3. ✅ **Documentation shows data sources** (this file!)
4. ✅ **README mentions real data would improve accuracy** (already done!)

You **don't need** to train on real data for submission - the multi-agent architecture is what's being evaluated!

---

## 📞 Data Source Support

**If downloads fail:**

1. **UNHCR:** Contact data@unhcr.org (usually responds in 1-2 days)
2. **ACLED:** Register at https://acleddata.com/register/ (instant access)
3. **World Bank:** All data is public API (no registration)
4. **NASA:** Register at https://urs.earthdata.nasa.gov/users/new (instant)

---

**Need help downloading? Run the automated script or let me know which dataset is giving you trouble!**
