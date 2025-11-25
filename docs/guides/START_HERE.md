# 🚀 START HERE - Quick Setup

## ✅ You Have All The Data!

Your datasets are ready:
- ✅ UNHCR Refugees: 693 rows
- ✅ ACLED Conflicts: 2,566 rows
- ✅ World Bank: 30 rows
- ✅ Climate: 600 rows

---

## 🎯 Three Simple Steps to Run Your AI System

### Step 1: Get Free Gemini API Key (2 minutes)

1. Visit: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (starts with `AIzaSy...`)
4. Open `.env` file in your project
5. Add: `GEMINI_API_KEY=AIzaSy_your_key_here`
6. Save the file

**Why?** The AI agents use Gemini for multi-modal analysis and coordination.

### Step 2: Train the Model (5-10 minutes)

```bash
python train.py --model lstm --epochs 100
```

**What happens?**
- Loads your UNHCR, ACLED, World Bank, and Climate data
- Trains LSTM neural network to predict refugee displacement
- Saves trained model to `models/trained/`
- Uses 20 features including conflict, economic, climate data

**Expected output:**
```
Epoch [10/100] - Train Loss: 0.0234, Val Loss: 0.0312
Epoch [20/100] - Train Loss: 0.0198, Val Loss: 0.0289
...
Best validation loss: 0.0156
Model saved to: models/trained/lstm_forecaster.pth
```

### Step 3: Run the Demo (30 seconds)

```bash
python main.py --mode demo
```

**What happens?**
- Runs 3 multi-agent workflows:
  1. **PARALLEL**: Vision + Forecasting run simultaneously
  2. **SEQUENTIAL**: Vision → Forecast → Resource → Communication
  3. **LOOPED**: Continuous refinement with new data
- Shows all 5 agents working together
- Demonstrates memory systems tracking predictions

**Expected output:**
```
[Vision Agent] Loaded TRAINED YOLO model
[Forecasting Agent] Loaded trained LSTM model
[Orchestrator] Starting PARALLEL workflow...
...
PARALLEL workflow completed: 24.14s
```

---

## 📚 Documentation Quick Links

| Document | What It Contains |
|----------|------------------|
| [DATA_READY.md](DATA_READY.md) | Your downloaded data summary |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Complete training documentation |
| [GET_API_KEY.md](GET_API_KEY.md) | How to get Gemini API key |
| [QUICK_COMMANDS.md](QUICK_COMMANDS.md) | Command reference |
| [README.md](README.md) | Full project overview |

---

## ⚡ Quick Commands

```bash
# Train LSTM (displacement forecasting)
python train.py --model lstm --epochs 100

# Run demo with all 3 workflows
python main.py --mode demo

# Check what data you have
python -c "import pandas as pd; print(pd.read_csv('data/unhcr_refugees_processed.csv').head())"

# View training results
cat models/trained/lstm_metadata.json
```

---

## 🎓 Understanding the System

### The 5 AI Agents

1. **Vision Intelligence** (YOLO + Gemini)
   - Analyzes satellite images for threats
   - Detects refugee camps, military vehicles, damaged buildings

2. **Displacement Forecasting** (LSTM)
   - Predicts refugee displacement 6 months ahead
   - Uses 20 features: conflict, economic, climate data
   - **This is what you're training with your data!**

3. **Resource Optimization**
   - Calculates humanitarian needs
   - Determines food, water, shelter, medical supplies

4. **Crisis Communication** (Gemini + Twilio)
   - Generates alerts in 4 languages
   - Sends SMS to aid organizations

5. **Orchestration & Debug** (Gemini)
   - Coordinates all agents
   - Validates outputs
   - Resolves conflicts

### The 3 Workflows

1. **PARALLEL**: Vision + Forecasting run at same time → faster
2. **SEQUENTIAL**: Each agent uses previous output → more accurate
3. **LOOPED**: Continuous refinement → adaptive to new data

### The 3 Memory Systems

1. **Conversation Memory**: Compares predictions vs actuals, learns accuracy
2. **Episodic Memory**: Logs all decisions to JSON for debugging
3. **Vector Memory**: Stores image embeddings, detects pattern changes

---

## 🔍 Troubleshooting

**Issue**: "No API_KEY or ADC found"
- **Solution**: Add Gemini API key to `.env` file (see Step 1)

**Issue**: "UNHCR data not found"
- **Solution**: Run `python process_my_data.py` to process your downloaded files

**Issue**: Training loss very high
- **Solution**: Normal for first few epochs. Loss decreases over time.

**Issue**: "ultralytics not installed"
- **Solution**: Run `pip install ultralytics` (optional, only needed for YOLO)

---

## 📊 What Your Data Contains

### UNHCR Refugees (693 rows)
```
Columns: Year, Country of Asylum, Country of Origin,
         Refugees, Asylum-seekers, IDPs, ...
```

### ACLED Conflicts (2,566 rows)
```
Columns: COUNTRY, YEAR, EVENTS
Example: Afghanistan, 2021, 9122 events
```

### World Bank (30 rows)
```
Columns: country, year, gdp_per_capita, population,
         inflation_rate, unemployment_rate, ...
```

### Climate (600 rows)
```
Columns: location, year, month, temperature_celsius,
         precipitation_mm, humidity_pct
```

---

## 🎯 Expected Results

After training with your data, the LSTM model will:
- ✅ Predict displacement based on conflict events
- ✅ Use economic indicators (GDP, unemployment) as features
- ✅ Incorporate climate factors (drought, floods)
- ✅ Learn patterns from 2014-2023 UNHCR data
- ✅ Forecast 6 months ahead with confidence intervals

**Validation loss** should decrease from ~10,000+ to <100 after 100 epochs.

---

## 🚀 Ready? Let's Go!

```bash
# 1. Add API key to .env
# 2. Train model
python train.py --model lstm --epochs 100

# 3. Run demo
python main.py --mode demo
```

**That's it!** Your AI-Powered Refugee Crisis Intelligence System will be running! 🎉

---

**Need help?** Check:
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training docs
- [README.md](README.md) for project overview
- [GET_API_KEY.md](GET_API_KEY.md) for API setup
