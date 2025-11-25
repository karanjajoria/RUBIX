"""
Configuration settings for the Refugee Crisis Intelligence System.
Centralizes all system parameters, API configurations, and model settings.
"""

import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Model Configuration
class ModelConfig:
    """Configuration for AI models used by different agents."""

    # Gemini Models
    GEMINI_PRO = "gemini-2.0-flash-exp"  # For vision and complex reasoning
    GEMINI_FLASH = "gemini-2.0-flash-exp"  # For fast coordination tasks

    # Vision Model (YOLO)
    YOLO_MODEL = "weights/yolov8n.pt"  # Lightweight model for demo
    YOLO_CONFIDENCE_THRESHOLD = 0.5
    YOLO_IOU_THRESHOLD = 0.45

    # Forecasting Model (LSTM)
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    FORECAST_HORIZON = 6  # months
    SEQUENCE_LENGTH = 12  # months of historical data


# Agent Configuration
class AgentConfig:
    """Configuration for multi-agent system."""

    # Agent names
    VISION_AGENT = "vision_intelligence"
    FORECASTING_AGENT = "displacement_forecasting"
    RESOURCE_AGENT = "resource_optimization"
    COMMUNICATION_AGENT = "crisis_communication"
    ORCHESTRATOR_AGENT = "orchestrator"

    # Timeout settings (seconds)
    AGENT_TIMEOUT = 120
    VISION_TIMEOUT = 60
    FORECASTING_TIMEOUT = 90

    # Memory settings
    MAX_MEMORY_ITEMS = 1000
    MEMORY_CLEANUP_THRESHOLD = 0.8


# Forecasting Configuration
class ForecastingConfig:
    """Configuration for displacement forecasting."""

    # Feature categories (90+ variables)
    FEATURES = {
        "conflict": [
            "conflict_events_count",
            "fatalities",
            "violence_against_civilians",
            "battles",
            "protests",
            "riots"
        ],
        "climate": [
            "temperature_avg",
            "precipitation",
            "drought_index",
            "flood_risk"
        ],
        "economic": [
            "gdp_per_capita",
            "food_price_index",
            "unemployment_rate",
            "inflation_rate"
        ],
        "demographic": [
            "population_density",
            "urban_population_pct",
            "age_dependency_ratio"
        ],
        "infrastructure": [
            "health_facilities_per_capita",
            "water_access_pct",
            "road_density"
        ]
    }

    # Threat levels
    THREAT_LEVELS = {
        "low": (0, 3),
        "medium": (3, 6),
        "high": (6, 8),
        "critical": (8, 10)
    }


# Vision Detection Configuration
class VisionConfig:
    """Configuration for vision intelligence."""

    # Detection classes
    THREAT_CLASSES = [
        "weapon",
        "military_vehicle",
        "tank",
        "artillery",
        "military_personnel",
        "damaged_building",
        "fire",
        "explosion"
    ]

    # Threat scoring weights
    THREAT_WEIGHTS = {
        "weapon": 2.0,
        "military_vehicle": 3.0,
        "tank": 4.0,
        "artillery": 4.5,
        "military_personnel": 1.5,
        "damaged_building": 2.5,
        "fire": 2.0,
        "explosion": 5.0
    }


# Communication Configuration
class CommunicationConfig:
    """Configuration for crisis communication."""

    # Alert priority levels
    PRIORITY_LEVELS = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "urgent": 4
    }

    # Supported languages for alerts
    SUPPORTED_LANGUAGES = [
        "en",  # English
        "fr",  # French
        "ar",  # Arabic
        "sw"   # Swahili
    ]

    # Alert templates
    ALERT_TEMPLATES = {
        "displacement_warning": {
            "en": "ALERT: Predicted displacement of {count} people in {location} within {months} months. Threat level: {level}.",
            "fr": "ALERTE: Déplacement prévu de {count} personnes à {location} dans {months} mois. Niveau de menace: {level}.",
        },
        "resource_need": {
            "en": "RESOURCE NEED: {location} requires {resources} to support {count} displaced persons.",
            "fr": "BESOIN EN RESSOURCES: {location} nécessite {resources} pour soutenir {count} personnes déplacées.",
        }
    }


# System Configuration
class SystemConfig:
    """General system configuration."""

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

    # Performance settings
    MAX_CONCURRENT_AGENTS = 5
    BATCH_SIZE = 32

    # Validation thresholds
    MIN_CONFIDENCE_SCORE = 0.6
    MAX_FALSE_POSITIVE_RATE = 0.1


# Export all configurations
__all__ = [
    "ModelConfig",
    "AgentConfig",
    "ForecastingConfig",
    "VisionConfig",
    "CommunicationConfig",
    "SystemConfig",
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "GEMINI_API_KEY"
]
