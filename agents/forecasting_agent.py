"""
Displacement Forecasting Agent
Predicts refugee displacement 4-6 months ahead using LSTM + 90+ variables.
Implements conversation memory to learn from predictions vs actual outcomes.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Warning: PyTorch or sklearn not installed. Forecasting features limited.")
    torch = None
    nn = None
    StandardScaler = None

import google.generativeai as genai

from config.config import (
    ModelConfig,
    ForecastingConfig,
    GEMINI_API_KEY,
    MODELS_DIR
)
from utils.memory import memory_manager
from utils.ollama_client import ollama_client


class LSTMForecaster(nn.Module if nn else object):
    """LSTM neural network for displacement forecasting."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        if nn is None:
            return

        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        if nn is None:
            return None

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class DisplacementForecastingAgent:
    """
    Agent 2: Displacement Forecasting
    Predicts refugee movements using historical data and 90+ variables.

    Multi-Agent Feature: PARALLEL processing with Vision Agent
    - Runs simultaneously with Vision Agent on different data streams
    - Uses Conversation Memory to improve predictions over time
    """

    def __init__(self):
        """Initialize Forecasting Agent with LSTM model and Gemini."""
        self.agent_name = "displacement_forecasting"

        # Configure Gemini Flash for fast trend summarization
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(ModelConfig.GEMINI_FLASH)

        # Feature configuration (90+ variables) - SET BEFORE initializing model
        self.feature_config = ForecastingConfig.FEATURES
        self.forecast_horizon = ModelConfig.FORECAST_HORIZON  # months

        # Initialize LSTM model
        self.lstm_model = None
        self.scaler = StandardScaler() if StandardScaler else None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize LSTM forecasting model."""
        if torch is None or nn is None:
            print("[Forecasting Agent] PyTorch not available. Using mock predictions.")
            return

        # Calculate total number of features
        total_features = sum(len(features) for features in self.feature_config.values())

        self.lstm_model = LSTMForecaster(
            input_size=total_features,
            hidden_size=ModelConfig.LSTM_HIDDEN_SIZE,
            num_layers=ModelConfig.LSTM_NUM_LAYERS,
            dropout=ModelConfig.LSTM_DROPOUT
        )

        # Load trained weights if available, otherwise use fresh model
        # Try real data model first, then fall back to synthetic
        trained_model_path = MODELS_DIR / "trained" / "lstm_forecaster_real.pth"
        scaler_path = MODELS_DIR / "trained" / "scaler_X_real.pkl"

        if not trained_model_path.exists():
            trained_model_path = MODELS_DIR / "trained" / "lstm_forecaster.pth"
            scaler_path = MODELS_DIR / "trained" / "scaler.pkl"

        if trained_model_path.exists():
            try:
                self.lstm_model.load_state_dict(torch.load(trained_model_path, map_location='cpu'))
                print(f"[Forecasting Agent] Loaded trained LSTM model from {trained_model_path}")

                # Load scaler if available
                if scaler_path.exists():
                    import pickle
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"[Forecasting Agent] Loaded trained scaler from {scaler_path}")

            except Exception as e:
                print(f"[Forecasting Agent] Error loading trained model: {e}. Using untrained model.")
        else:
            print(f"[Forecasting Agent] No trained model found. Using fresh LSTM model.")
            print(f"[Forecasting Agent] Train with: python train.py --model lstm --epochs 100")
        # For demo, use untrained model with mock predictions

    def forecast_displacement(self, location: str, historical_data: pd.DataFrame,
                             threat_level: Optional[str] = None,
                             vision_threat_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Forecast displacement for a location.

        Sequential Workflow Integration:
        - Takes threat_level from Vision Agent as input
        - Produces forecast that feeds into Resource Agent

        Args:
            location: Geographic location (lat, lon)
            historical_data: DataFrame with 90+ features (conflict, climate, economic, etc.)
            threat_level: Current threat level from Vision Agent
            vision_threat_score: Numeric threat score from Vision Agent

        Returns:
            Dictionary containing:
                - predicted_displacement: Number of people expected to displace
                - forecast_months: List of monthly predictions
                - confidence_interval: 95% CI bounds
                - risk_factors: Key factors driving displacement
                - trend_summary: Gemini-generated trend analysis
        """
        start_time = datetime.now()

        try:
            # Step 1: Prepare features from historical data
            features = self._prepare_features(historical_data)

            # Step 2: Incorporate threat level from Vision Agent (Context Engineering)
            # NOTE: Vision integration disabled - model trained without this feature
            # if vision_threat_score is not None:
            #     features = self._integrate_vision_context(features, vision_threat_score)

            # Step 3: Generate LSTM forecast
            predictions = self._lstm_forecast(features)

            # Step 4: Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(predictions)

            # Step 5: Identify key risk factors
            risk_factors = self._identify_risk_factors(historical_data, vision_threat_score)

            # Step 6: Gemini trend summarization
            trend_summary = self._gemini_summarize_trend(
                location=location,
                predictions=predictions,
                risk_factors=risk_factors,
                threat_level=threat_level
            )

            # Step 7: Store prediction in conversation memory
            total_displacement = int(sum(predictions))
            memory_manager.conversation_memory.add_prediction(
                timestamp=datetime.now().isoformat(),
                location=location,
                predicted_displacement=total_displacement,
                confidence=0.85,  # Model confidence (would be calculated from validation)
                metadata={
                    "threat_level": threat_level,
                    "vision_threat_score": vision_threat_score,
                    "forecast_horizon": self.forecast_horizon
                }
            )

            # Prepare result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = {
                "predicted_displacement": total_displacement,
                "forecast_months": predictions.tolist(),
                "confidence_interval_95": confidence_intervals,
                "risk_factors": risk_factors,
                "trend_summary": trend_summary,
                "location": location,
                "forecast_horizon_months": self.forecast_horizon,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time
            }

            # Log to episodic memory
            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="forecast_displacement",
                input_data={"location": location, "threat_level": threat_level},
                output_data=result,
                status="success",
                metadata={"execution_time": execution_time}
            )

            return result

        except Exception as e:
            # Log error to episodic memory
            error_result = {
                "error": str(e),
                "location": location
            }

            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="forecast_displacement",
                input_data={"location": location, "threat_level": threat_level},
                output_data=error_result,
                status="error",
                metadata={"error_message": str(e)}
            )

            return error_result

    def _prepare_features(self, historical_data: pd.DataFrame) -> np.ndarray:
        """
        Extract and scale 90+ features from historical data.

        Feature Categories:
        - Conflict: event counts, fatalities, violence types
        - Climate: temperature, precipitation, drought indices
        - Economic: GDP, food prices, unemployment
        - Demographic: population density, urbanization
        - Infrastructure: health facilities, water access
        """
        # Extract all feature columns
        all_features = []
        for category, feature_names in self.feature_config.items():
            for feature in feature_names:
                if feature in historical_data.columns:
                    all_features.append(historical_data[feature].values)
                else:
                    # Fill missing features with zeros (in production, use proper imputation)
                    all_features.append(np.zeros(len(historical_data)))

        # Stack into matrix
        feature_matrix = np.column_stack(all_features)

        # Scale features
        if self.scaler is not None:
            feature_matrix = self.scaler.fit_transform(feature_matrix)

        return feature_matrix

    def _integrate_vision_context(self, features: np.ndarray, threat_score: float) -> np.ndarray:
        """
        Integrate Vision Agent's threat score as additional context.

        Context Engineering: Dynamic feature augmentation
        """
        # Add threat score as additional feature column
        threat_feature = np.full((features.shape[0], 1), threat_score / 10.0)  # Normalize to 0-1
        return np.hstack([features, threat_feature])

    def _lstm_forecast(self, features: np.ndarray) -> np.ndarray:
        """
        Generate LSTM forecast for displacement.

        Returns array of monthly predictions for forecast horizon.
        """
        if self.lstm_model is None or torch is None:
            # Mock predictions for demo
            return self._mock_forecast()

        try:
            # Prepare sequence (last N months as input)
            sequence_length = ModelConfig.SEQUENCE_LENGTH
            if len(features) < sequence_length:
                # Pad if insufficient history
                features = np.pad(features, ((sequence_length - len(features), 0), (0, 0)), mode='edge')

            sequence = features[-sequence_length:]

            # Convert to tensor
            x = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension

            # Forecast
            self.lstm_model.eval()
            with torch.no_grad():
                predictions = []
                current_seq = x

                for _ in range(self.forecast_horizon):
                    pred = self.lstm_model(current_seq)
                    predictions.append(pred.item())

                    # Shift sequence for next prediction
                    # (simplified - in production, update with predicted features)
                    # pred is shape (batch, 1), expand to (batch, 1, features)
                    pred_expanded = pred.unsqueeze(1).expand(-1, 1, current_seq.size(2))
                    current_seq = torch.cat([current_seq[:, 1:, :], pred_expanded], dim=1)

            return np.array(predictions)

        except Exception as e:
            print(f"[Forecasting Agent] LSTM forecast error: {e}")
            return self._mock_forecast()

    def _mock_forecast(self) -> np.ndarray:
        """Generate mock forecast for demo purposes."""
        # Simulate increasing displacement trend
        base = 500
        predictions = [base + np.random.randint(-100, 200) for _ in range(self.forecast_horizon)]
        return np.array(predictions)

    def _calculate_confidence_intervals(self, predictions: np.ndarray, confidence: float = 0.95) -> List[Tuple[float, float]]:
        """
        Calculate confidence intervals for predictions.

        Uses historical accuracy from conversation memory.
        """
        # Get historical accuracy metrics
        accuracy = memory_manager.conversation_memory.get_accuracy_metrics()
        rmse = accuracy.get("rmse", 100)  # Default RMSE if no history

        # Calculate 95% CI (±1.96 * RMSE)
        z_score = 1.96
        margin = z_score * rmse

        confidence_intervals = [
            (max(0, pred - margin), pred + margin)
            for pred in predictions
        ]

        return confidence_intervals

    def _identify_risk_factors(self, historical_data: pd.DataFrame, vision_threat_score: Optional[float]) -> List[Dict[str, Any]]:
        """
        Identify key factors driving displacement risk.

        Returns top risk factors with their contribution.
        """
        risk_factors = []

        # Conflict intensity
        if 'conflict_events_count' in historical_data.columns:
            recent_conflicts = historical_data['conflict_events_count'].tail(3).mean()
            if recent_conflicts > 10:
                risk_factors.append({
                    "factor": "High conflict intensity",
                    "value": recent_conflicts,
                    "weight": 0.4
                })

        # Vision threat score
        if vision_threat_score and vision_threat_score > 6:
            risk_factors.append({
                "factor": "Detected military threats",
                "value": vision_threat_score,
                "weight": 0.35
            })

        # Food prices
        if 'food_price_index' in historical_data.columns:
            recent_prices = historical_data['food_price_index'].tail(3).mean()
            baseline = historical_data['food_price_index'].mean()
            if recent_prices > baseline * 1.2:
                risk_factors.append({
                    "factor": "Rising food prices",
                    "value": recent_prices,
                    "weight": 0.15
                })

        # Climate stress
        if 'drought_index' in historical_data.columns:
            drought_severity = historical_data['drought_index'].tail(1).values[0]
            if drought_severity > 0.6:
                risk_factors.append({
                    "factor": "Climate stress (drought)",
                    "value": drought_severity,
                    "weight": 0.1
                })

        return sorted(risk_factors, key=lambda x: x['weight'], reverse=True)

    def _gemini_summarize_trend(self, location: str, predictions: np.ndarray,
                               risk_factors: List[Dict], threat_level: Optional[str]) -> str:
        """
        Use Gemini Flash for fast trend summarization.

        Context Engineering: Specialized prompt for humanitarian forecasting
        """
        # Prepare trend data
        trend_direction = "increasing" if predictions[-1] > predictions[0] else "decreasing"
        total_predicted = int(sum(predictions))
        peak_month = int(np.argmax(predictions)) + 1

        # Prepare risk factors summary
        risk_summary = "\n".join([
            f"- {rf['factor']}: {rf['value']:.1f} (weight: {rf['weight']:.0%})"
            for rf in risk_factors[:3]
        ]) if risk_factors else "No significant risk factors identified"

        # Specialized prompt for Forecasting Agent
        prompt = f"""You are a humanitarian AI forecasting refugee displacement patterns.

Location: {location}
Current Threat Level: {threat_level or 'Unknown'}
Forecast Horizon: {self.forecast_horizon} months

Predicted Displacement:
- Total: {total_predicted:,} people
- Trend: {trend_direction}
- Peak expected: Month {peak_month}

Key Risk Factors:
{risk_summary}

Task: Provide a concise trend summary (2-3 sentences) for humanitarian planners:
1. Overall displacement trajectory
2. Critical factors driving movement
3. Recommended preparedness timeline"""

        try:
            import time
            max_retries = 3
            retry_delay = 2  # seconds

            # Try Gemini first
            for attempt in range(max_retries):
                try:
                    response = self.gemini_model.generate_content(prompt)
                    return response.text.strip()
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"[Forecasting Agent] Gemini quota limit hit, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"[Forecasting Agent] Gemini quota exhausted, switching to Ollama (Llama 3)...")
                            break
                    elif "API_KEY" in str(e) or "ADC" in str(e):
                        print(f"[Forecasting Agent] Gemini API key not configured, using Ollama (Llama 3)...")
                        break
                    raise e

            # Fallback to Ollama (Llama 3)
            if ollama_client.is_available():
                print(f"[Forecasting Agent] Using Ollama (Llama 3) for trend summarization...")
                ollama_response = ollama_client.generate(
                    prompt=prompt,
                    system="You are a humanitarian AI assistant specializing in refugee displacement forecasting.",
                    temperature=0.7,
                    max_tokens=300
                )
                if ollama_response:
                    return ollama_response

            # Final fallback if both fail
            return f"Forecasting {total_predicted:,} displaced persons over {self.forecast_horizon} months with {trend_direction} trend."

        except Exception as e:
            print(f"[Forecasting Agent] LLM summarization error: {e}")
            return f"Forecasting {total_predicted:,} displaced persons over {self.forecast_horizon} months with {trend_direction} trend."

    def update_with_actual(self, location: str, timestamp: str, actual_displacement: int):
        """
        Update conversation memory with actual displacement outcomes.

        Enables continuous learning and accuracy improvement.
        """
        memory_manager.conversation_memory.add_actual(
            timestamp=timestamp,
            location=location,
            actual_displacement=actual_displacement,
            metadata={"source": "ground_truth_validation"}
        )

        # Log memory update
        memory_manager.episodic_memory.log_episode(
            agent_name=self.agent_name,
            action="update_with_actual",
            input_data={"location": location, "actual_displacement": actual_displacement},
            output_data={"status": "memory_updated"},
            status="success",
            metadata={}
        )


# Create global instance
forecasting_agent = DisplacementForecastingAgent()
