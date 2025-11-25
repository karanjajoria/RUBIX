# Web Interface - AI-Powered Refugee Crisis Intelligence System

Interactive web dashboard to showcase the multi-agent system with live demonstrations and visualizations.

## 🌐 Features

### 1. **System Dashboard**
- Real-time system status and health monitoring
- Live metrics (predictions, accuracy, threats detected)
- LLM backend status (Gemini or Llama 3)
- Agent operational status

### 2. **Agent Showcase**
- Visual cards for all 5 specialized agents
- Agent capabilities and roles
- Status indicators
- Interactive hover effects

### 3. **Workflow Execution**
- **Parallel Workflow**: Run Vision + Forecasting simultaneously
- **Sequential Workflow**: Complete pipeline (Vision → Forecast → Resource → Communication)
- **Looped Workflow**: 3 iterations with continuous refinement
- Real-time output streaming
- Live demo console with color-coded logs

### 4. **Data Visualizations**
- **Predictions Chart**: LSTM predictions vs actual displacement (line chart)
- **Threats Map**: Top crisis regions by threat level (bar chart)
- **Model Metrics**: Training/validation loss, accuracy
- **System Architecture**: Visual representation of components

### 5. **Competition Showcase**
- Project scorecard with animated progress bars
- Technical achievements
- Data pipeline overview
- Deployment information

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd web
pip install flask
```

(Flask is already in requirements.txt if you install from root)

### 2. Run the Web Server

```bash
python app.py
```

The server will start at: **http://localhost:5000**

### 3. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 File Structure

```
web/
├── app.py                      # Flask backend API
├── templates/
│   └── index.html              # Main dashboard page
├── static/
│   ├── css/
│   │   └── style.css           # Styles and animations
│   └── js/
│       └── main.js             # Frontend logic and charts
└── README.md                   # This file
```

## 🎯 API Endpoints

### System Information
- `GET /api/status` - Get system status
- `GET /api/metrics` - Get performance metrics
- `GET /api/agents` - Get agent information
- `GET /api/workflows` - Get workflow definitions

### Demo Execution
- `POST /api/demo/run` - Start a workflow demo
  ```json
  {
    "workflow": "parallel" | "sequential" | "looped"
  }
  ```
- `GET /api/demo/stream` - Stream demo output (Server-Sent Events)

### Data Visualization
- `GET /api/data/predictions` - Get prediction time series
- `GET /api/data/countries` - Get country-level threat data

## 🎨 Features Showcase

### Interactive Charts
Uses **Chart.js** for beautiful, responsive visualizations:
- Line charts for time series predictions
- Bar charts for threat levels
- Animated transitions
- Interactive tooltips

### Live Demo Execution
- Click any workflow card to run a demo
- Real-time console output with color coding:
  - 🟢 Green: Success messages
  - 🔵 Blue: Results and data
  - ⚪ Gray: Logs and info
  - 🔴 Red: Errors
- Smooth animations and transitions

### Responsive Design
- Mobile-friendly layout
- Adaptive grid system
- Touch-optimized controls
- Works on all screen sizes

## 🔧 Configuration

### Environment Variables

The web interface uses the same `.env` file as the main system:

```bash
# Required for AI reasoning
GEMINI_API_KEY=your_gemini_api_key_here

# Optional for full functionality
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
```

### Port Configuration

Default port is **5000**. To change it, edit `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

## 🎬 Demo Workflows

### Parallel Workflow (~24-81s)
```
Vision Agent     ┐
                 ├─→ Simultaneous Execution
Forecasting      ┘
```

### Sequential Workflow (~60-120s)
```
Vision → Forecasting → Resource → Communication
```

### Looped Workflow (~90-150s)
```
Iteration 1: Vision + Forecasting
Iteration 2: Vision + Forecasting (refined)
Iteration 3: Vision + Forecasting (final)
```

## 📊 Metrics Displayed

| Metric | Description |
|--------|-------------|
| **Predictions Made** | Total displacement predictions |
| **Avg Accuracy** | Prediction accuracy percentage |
| **Threats Detected** | Visual threats identified |
| **Countries Monitored** | Countries in training data |
| **Training Loss** | LSTM model training loss |
| **Validation Loss** | LSTM model validation loss |

## 🎯 Use Cases

### 1. **Project Demonstrations**
- Show to competition judges
- Present at conferences
- Share with stakeholders
- Portfolio showcase

### 2. **Development & Testing**
- Visual debugging of agent workflows
- Real-time monitoring during development
- Performance benchmarking
- System health checks

### 3. **Educational Purposes**
- Teaching multi-agent systems
- ML model deployment demonstrations
- Web development examples
- API design patterns

## 🛠️ Troubleshooting

### Issue: Port Already in Use
```bash
# Kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Or change port in app.py
```

### Issue: Charts Not Displaying
- Ensure Chart.js CDN is accessible
- Check browser console for errors
- Verify API endpoints are responding

### Issue: Demo Not Running
- Check that main system dependencies are installed
- Verify agents are importable
- Check `.env` file for API keys
- Look for errors in Flask console

## 🚀 Deployment Options

### Option 1: Local Development
```bash
python app.py
```

### Option 2: Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 3: Docker Container
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "web/app.py"]
```

### Option 4: Cloud Deployment
- **Google Cloud Run**: Use included `cloudbuild.yaml`
- **Heroku**: Add `Procfile` with `web: python web/app.py`
- **AWS Elastic Beanstalk**: Deploy Flask app directly

## 📝 Customization

### Adding New Visualizations

1. Add API endpoint in `app.py`:
```python
@app.route('/api/data/custom')
def api_custom_data():
    return jsonify({'data': [...]})
```

2. Add chart in `main.js`:
```javascript
async function initializeCustomChart() {
    const response = await fetch('/api/data/custom');
    const data = await response.json();
    // Create chart...
}
```

3. Add canvas in `index.html`:
```html
<canvas id="custom-chart"></canvas>
```

### Changing Color Scheme

Edit CSS variables in `style.css`:
```css
:root {
    --primary: #2563eb;      /* Change primary color */
    --secondary: #64748b;    /* Change secondary color */
    --success: #10b981;      /* Change success color */
}
```

## 🏆 Competition Presentation

Perfect for showcasing your project:

1. **Open the dashboard** before the presentation
2. **Explain the architecture** using the agent cards
3. **Run a live demo** to show real-time execution
4. **Show visualizations** to demonstrate model performance
5. **Highlight metrics** to prove system effectiveness

## 📚 Additional Resources

- **Main Documentation**: See [../README.md](../README.md)
- **Quick Start Guide**: See [../docs/guides/QUICK_START.md](../docs/guides/QUICK_START.md)
- **Technical Details**: See [../docs/technical/](../docs/technical/)

## ✅ Status

✅ **Backend API** - All endpoints functional
✅ **Frontend UI** - Responsive and interactive
✅ **Live Demos** - Real-time workflow execution
✅ **Visualizations** - Charts and metrics working
✅ **Documentation** - Complete and detailed

## 🎉 Ready to Showcase!

Your web interface is production-ready and perfect for:
- 🏆 Competition presentations
- 📊 Live demonstrations
- 🎓 Educational purposes
- 💼 Portfolio showcase

**Access at**: http://localhost:5000

---

**Last Updated**: November 23, 2024
**Status**: ✅ Production Ready
**Version**: 1.0.0
