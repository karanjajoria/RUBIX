# Troubleshooting Guide

Common issues and solutions for the AI-Powered Refugee Crisis Intelligence System.

---

## Training Errors

### ✅ FIXED: "ReduceLROnPlateau got unexpected keyword argument 'verbose'"

**Issue**: PyTorch deprecated the `verbose` parameter in newer versions.

**Solution**: Already fixed in train.py (removed verbose parameter).

**Verification**:
```bash
python train.py --model lstm --epochs 5
# Should complete without errors
```

---

### "No API_KEY or ADC found"

**Issue**: Gemini API key not configured.

**Solution**:
1. Get free key: https://aistudio.google.com/app/apikey
2. Open `.env` file
3. Add: `GEMINI_API_KEY=your_key_here`
4. Save file

---

### "UNHCR data not found"

**Issue**: Training script looking for specific filename.

**Solution**: This is normal! The script automatically generates synthetic data if real data is not in the expected location.

**To use your real data**:
```bash
# Option 1: Already processed (recommended)
# Your data is in: data/unhcr_refugees_processed.csv
# Train.py uses synthetic data by default - this is fine for testing!

# Option 2: Rename your processed file (if you want to use real data)
cp data/unhcr_refugees_processed.csv data/unhcr_uganda.csv
python train.py --model lstm --epochs 100
```

---

### Training loss is very high (>1,000,000)

**Status**: Normal for synthetic data and early epochs.

**Explanation**:
- Synthetic displacement data has large values (50,000-100,000 people)
- MSE loss = (prediction - actual)²
- Early predictions are random, so errors are huge
- Loss will decrease over epochs

**What to do**: Train for more epochs (100+)
```bash
python train.py --model lstm --epochs 100
```

**Expected behavior**:
```
Epoch [10/100] - Train Loss: 15000000, Val Loss: 14000000
Epoch [20/100] - Train Loss: 5000000,  Val Loss: 6000000
Epoch [50/100] - Train Loss: 500000,   Val Loss: 800000
Epoch [100/100] - Train Loss: 50000,   Val Loss: 100000
```

---

## Installation Errors

### "ultralytics not installed"

**Issue**: YOLO package missing (optional).

**Solution**:
```bash
pip install ultralytics
```

**Or**: Skip YOLO training (not required for LSTM)
```bash
python train.py --model lstm --epochs 100  # Works without ultralytics
```

---

### "openpyxl not installed"

**Issue**: Excel file support missing.

**Solution**:
```bash
pip install openpyxl
```

**Verification**: Already installed if you ran `process_my_data.py` successfully.

---

### "Module not found" errors

**Issue**: Dependencies not installed.

**Solution**:
```bash
# Full installation
pip install -r requirements.txt

# Or simplified
pip install -r scripts/requirements_simplified.txt

# Windows batch
scripts\install.bat
```

---

## Data Errors

### "Cannot read CSV/Excel file"

**Issue**: File encoding or format issues.

**Solutions**:

1. **Check file exists**:
   ```bash
   ls data/
   ```

2. **Try different encoding**:
   ```python
   # In Python
   import pandas as pd
   df = pd.read_csv('data/your_file.csv', encoding='utf-8')
   # Or try: encoding='latin-1', encoding='cp1252'
   ```

3. **Use processing script**:
   ```bash
   python scripts/process_my_data.py
   ```

---

### "No data downloaded"

**Issue**: Network issues or API limits.

**Solution**:
```bash
# Re-run download scripts
python scripts/download_remaining_data.py

# Process your manual downloads
python scripts/process_my_data.py
```

---

## Runtime Errors

### "CUDA out of memory"

**Issue**: GPU memory full (only if using GPU).

**Solutions**:

1. **Use CPU** (automatic fallback):
   ```python
   # train.py already uses device='cpu'
   python train.py --model lstm --epochs 100
   ```

2. **Reduce batch size**:
   ```bash
   python train.py --model lstm --batch-size 16  # Default is 32
   ```

3. **Clear GPU cache** (if using GPU):
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### "KeyError: 'execution_time'"

**Issue**: Old version of orchestrator_agent.py.

**Status**: Already fixed!

**Verification**: Check that `orchestrator_agent.py` includes `execution_time` in error results (line 546).

---

### Demo hangs or takes forever

**Issue**: Large models or slow operations.

**Solutions**:

1. **Check API key**: Missing Gemini key causes delays
2. **Reduce operations**: Use smaller batch in demo
3. **Be patient**: First run downloads YOLO model (~6 MB)

---

## Performance Issues

### Training is very slow

**Expected times** (on CPU):
- 5 epochs: ~10 seconds
- 100 epochs: ~2-5 minutes
- YOLO (50 epochs): ~1-2 hours

**Speed up**:

1. **Reduce batch size** (paradoxically can be faster on CPU):
   ```bash
   python train.py --model lstm --batch-size 16
   ```

2. **Use GPU** (if available):
   ```python
   # Edit train.py, change device='cpu' to device='cuda'
   ```

3. **Reduce data**:
   - Training already uses synthetic data (120 months)
   - This is optimal for speed vs accuracy

---

## File Organization Issues

### "Cannot find docs/TRAINING_GUIDE.md"

**Issue**: Files were reorganized.

**Solution**: Documentation moved to `docs/` folder.

**New paths**:
```
docs/TRAINING_GUIDE.md      (was TRAINING_GUIDE.md)
docs/DATA_READY.md          (was DATA_READY.md)
scripts/download_data.py    (was download_data.py)
scripts/deployment/deploy.sh (was deploy.sh)
```

---

### "No such file or directory"

**Issue**: Running script from wrong directory.

**Solution**: Always run from project root:
```bash
# Check you're in the right place
pwd  # Should show: .../Google-Kaggle

ls   # Should show: agents/, config/, data/, docs/, scripts/, main.py, train.py

# Then run commands
python train.py --model lstm --epochs 100
python main.py --mode demo
```

---

## Model Loading Issues

### "Cannot load trained model"

**Issue**: Model file corrupted or incompatible.

**Solutions**:

1. **Retrain model**:
   ```bash
   rm models/trained/lstm_forecaster.pth
   python train.py --model lstm --epochs 100
   ```

2. **Check file exists**:
   ```bash
   ls models/trained/
   # Should show: lstm_forecaster.pth, scaler.pkl, lstm_metadata.json
   ```

3. **Use fresh model**: Agents fall back to untrained model if file missing.

---

### "Model architecture mismatch"

**Issue**: Trained model has different architecture than code.

**Solution**: Retrain with current code:
```bash
rm models/trained/*.pth
python train.py --model lstm --epochs 100
```

---

## API Issues

### "Gemini quota exceeded (429)"

**Issue**: Free tier rate limit (60 requests/minute).

**Solutions**:

1. **Wait 1 minute**: Quota resets
2. **Reduce parallel calls**: Demo already optimized
3. **Upgrade**: Get paid API key (optional)

---

### "Network connection failed"

**Issue**: Internet connectivity or firewall.

**Solutions**:

1. **Check internet**: `ping google.com`
2. **Check proxy**: If behind corporate firewall
3. **Use synthetic data**: Train without downloads
   ```bash
   python train.py --model lstm --epochs 100
   # Uses synthetic data automatically
   ```

---

## Validation Errors

### "Validation loss increases"

**Status**: Can be normal.

**Causes**:
- Overfitting (model memorizing training data)
- Learning rate too high
- Not enough training data

**Solutions**:

1. **Train longer**: Model may recover
2. **Reduce learning rate**:
   ```bash
   python train.py --model lstm --learning-rate 0.0005
   ```
3. **Add more data**: Use real UNHCR data

---

## Windows-Specific Issues

### "Permission denied" errors

**Solution**: Run terminal as Administrator
```bash
# Right-click Command Prompt -> Run as Administrator
```

---

### Path errors with backslashes

**Issue**: Windows uses `\`, Python uses `/`.

**Solution**: Use forward slashes or raw strings:
```python
# Good
path = "data/file.csv"
path = r"data\file.csv"  # raw string

# Bad
path = "data\file.csv"  # \f is interpreted as escape
```

---

## Getting Help

### Still having issues?

1. **Check logs**:
   ```bash
   cat logs/episodic_memory_*.json  # Agent decisions
   ```

2. **Read documentation**:
   - [START_HERE.md](../START_HERE.md)
   - [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
   - [QUICK_COMMANDS.md](QUICK_COMMANDS.md)

3. **Verify setup**:
   ```bash
   # Check Python version (should be 3.10+)
   python --version

   # Check packages
   pip list | grep torch
   pip list | grep pandas

   # Check data
   ls data/
   ```

4. **Clean reinstall**:
   ```bash
   # Remove virtual environment
   rm -rf env/

   # Recreate
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate     # Windows

   # Reinstall
   pip install -r requirements.txt
   ```

---

## Quick Diagnostics

Run this to check system health:

```bash
# Check Python
python --version

# Check packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# Check data
ls data/*.csv

# Check models
ls models/trained/

# Test import
python -c "from agents.forecasting_agent import forecasting_agent; print('Agents OK')"

# Test training (5 epochs, ~10 seconds)
python train.py --model lstm --epochs 5
```

---

**Last Updated**: November 23, 2024
