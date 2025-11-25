"""
Vision Intelligence Agent
Detects conflict threats from satellite/drone imagery using YOLO + Gemini 2.5 Pro.
Implements multi-modal reasoning for context-aware threat assessment.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

try:
    import cv2
    from ultralytics import YOLO
except ImportError:
    print("Warning: CV2 or Ultralytics not installed. Vision features limited.")
    cv2 = None
    YOLO = None

import google.generativeai as genai

from config.config import (
    ModelConfig,
    VisionConfig,
    GEMINI_API_KEY,
    MODELS_DIR
)
from utils.memory import memory_manager
from utils.ollama_client import ollama_client


class VisionIntelligenceAgent:
    """
    Agent 1: Vision Intelligence
    Analyzes satellite/drone imagery to detect conflict threats.

    Multi-Agent Feature: PARALLEL processing with Forecasting Agent
    - Vision Agent analyzes satellite feeds simultaneously while
      Forecasting Agent processes historical data
    """

    def __init__(self):
        """Initialize Vision Agent with YOLO and Gemini models."""
        self.agent_name = "vision_intelligence"

        # Configure Gemini for multi-modal reasoning
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(ModelConfig.GEMINI_PRO)

        # Load YOLO model (prefer trained model if available)
        self.yolo_model = None
        if YOLO is not None:
            try:
                # First, check for trained custom model
                custom_model_path = MODELS_DIR / "trained" / "yolo_conflict_custom.pt"

                if custom_model_path.exists():
                    self.yolo_model = YOLO(str(custom_model_path))
                    print(f"[Vision Agent] Loaded TRAINED YOLO model: {custom_model_path}")
                else:
                    # Fallback to pretrained YOLOv8n
                    model_path = MODELS_DIR / ModelConfig.YOLO_MODEL
                    if model_path.exists():
                        self.yolo_model = YOLO(str(model_path))
                        print(f"[Vision Agent] Loaded pretrained YOLO: {model_path}")
                    else:
                        # Download pretrained model if not exists
                        self.yolo_model = YOLO(ModelConfig.YOLO_MODEL)
                        print(f"[Vision Agent] Downloaded YOLO model: {ModelConfig.YOLO_MODEL}")

                    print(f"[Vision Agent] No trained model found. Train with: python train.py --model yolo --epochs 50")
            except Exception as e:
                print(f"[Vision Agent] Error loading YOLO: {e}")

        # Detection configuration
        self.confidence_threshold = ModelConfig.YOLO_CONFIDENCE_THRESHOLD
        self.threat_classes = VisionConfig.THREAT_CLASSES
        self.threat_weights = VisionConfig.THREAT_WEIGHTS

    def analyze_image(self, image_path: str, location: str, timestamp: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze satellite/drone image for conflict threats.

        Args:
            image_path: Path to the image file
            location: Geographic location (lat, lon) as string
            timestamp: Image capture timestamp (ISO format)
            context: Additional context (weather, recent events, etc.)

        Returns:
            Dictionary containing:
                - detections: List of detected objects
                - threat_score: Overall threat level (0-10)
                - threat_level: Category (low/medium/high/critical)
                - gemini_analysis: Multi-modal reasoning from Gemini
                - embedding: Image embedding for vector memory
        """
        start_time = datetime.now()

        try:
            # Read image
            if cv2 is None:
                raise ImportError("OpenCV not installed")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Step 1: YOLO object detection
            detections = self._yolo_detect(image)

            # Step 2: Calculate threat score
            threat_score = self._calculate_threat_score(detections)
            threat_level = self._classify_threat_level(threat_score)

            # Step 3: Gemini multi-modal reasoning
            gemini_analysis = self._gemini_analyze(
                image_path=image_path,
                detections=detections,
                location=location,
                context=context or {}
            )

            # Step 4: Generate image embedding for vector memory
            embedding = self._generate_embedding(image)

            # Step 5: Store in vector memory for pattern detection
            memory_manager.vector_memory.add_embedding(
                embedding=embedding,
                location=location,
                timestamp=timestamp or datetime.now().isoformat(),
                detections=[d["class"] for d in detections],
                threat_score=threat_score
            )

            # Log to episodic memory
            execution_time = (datetime.now() - start_time).total_seconds()
            result = {
                "detections": detections,
                "threat_score": threat_score,
                "threat_level": threat_level,
                "gemini_analysis": gemini_analysis,
                "location": location,
                "timestamp": timestamp or datetime.now().isoformat(),
                "execution_time": execution_time
            }

            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="analyze_image",
                input_data={"image_path": image_path, "location": location},
                output_data=result,
                status="success",
                metadata={"execution_time": execution_time}
            )

            return result

        except Exception as e:
            # Log error to episodic memory
            error_result = {
                "error": str(e),
                "image_path": image_path,
                "location": location
            }

            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="analyze_image",
                input_data={"image_path": image_path, "location": location},
                output_data=error_result,
                status="error",
                metadata={"error_message": str(e)}
            )

            return error_result

    def _yolo_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform YOLO object detection.

        Returns:
            List of detections with class, confidence, bounding box
        """
        if self.yolo_model is None:
            # Return mock detections for demo if YOLO not available
            return self._mock_detections()

        try:
            results = self.yolo_model(image, conf=self.confidence_threshold)
            detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()

                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    })

            return detections

        except Exception as e:
            print(f"[Vision Agent] YOLO detection error: {e}")
            return self._mock_detections()

    def _mock_detections(self) -> List[Dict[str, Any]]:
        """Generate mock detections for demo purposes."""
        return [
            {"class": "military_vehicle", "confidence": 0.87, "bbox": [100, 150, 250, 300]},
            {"class": "weapon", "confidence": 0.92, "bbox": [400, 200, 500, 350]},
            {"class": "weapon", "confidence": 0.78, "bbox": [600, 100, 700, 250]}
        ]

    def _calculate_threat_score(self, detections: List[Dict[str, Any]]) -> float:
        """
        Calculate overall threat score (0-10) based on detections.

        Uses weighted scoring based on threat class importance.
        """
        if not detections:
            return 0.0

        total_score = 0.0
        for detection in detections:
            class_name = detection["class"]
            confidence = detection["confidence"]

            # Get weight for this threat class (default to 1.0)
            weight = self.threat_weights.get(class_name, 1.0)

            # Score = weight * confidence
            total_score += weight * confidence

        # Normalize to 0-10 scale (assuming max ~5 high-threat detections)
        normalized_score = min(10.0, (total_score / 5.0) * 10.0)

        return round(normalized_score, 2)

    def _classify_threat_level(self, threat_score: float) -> str:
        """Classify threat score into categorical level."""
        for level, (min_score, max_score) in {
            "critical": (8, 10),
            "high": (6, 8),
            "medium": (3, 6),
            "low": (0, 3)
        }.items():
            if min_score <= threat_score < max_score:
                return level
        return "low"

    def _gemini_analyze(self, image_path: str, detections: List[Dict[str, Any]],
                       location: str, context: Dict[str, Any]) -> str:
        """
        Use Gemini 2.5 Pro for multi-modal reasoning about the image.

        Context Engineering: Specialized prompt for conflict assessment
        """
        # Prepare detection summary
        detection_summary = ", ".join([
            f"{d['confidence']:.0%} confident {d['class']}"
            for d in detections
        ]) if detections else "No significant threats detected"

        # Prepare context information
        context_info = "\n".join([
            f"- {k}: {v}" for k, v in context.items()
        ]) if context else "No additional context available"

        # Specialized prompt for Vision Agent
        prompt = f"""You are a humanitarian AI analyzing satellite imagery for conflict threat assessment.

Location: {location}
Detections from object detection model: {detection_summary}

Additional Context:
{context_info}

Task: Assess the conflict escalation risk considering:
1. Detected military equipment/weapons
2. Geographic context and recent events
3. Potential for civilian displacement
4. Urgency of humanitarian response

Provide a concise assessment (2-3 sentences) focusing on:
- Immediate threats to civilians
- Likelihood of displacement
- Recommended alert priority (low/medium/high/urgent)"""

        try:
            import time
            max_retries = 3
            retry_delay = 2  # seconds

            # Multi-modal: pass both image and text prompt
            with open(image_path, 'rb') as img_file:
                image_data = img_file.read()

            # Try Gemini first
            for attempt in range(max_retries):
                try:
                    response = self.gemini_model.generate_content([
                        prompt,
                        {"mime_type": "image/jpeg", "data": image_data}
                    ])
                    return response.text.strip()
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"[Vision Agent] Gemini quota limit hit, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"[Vision Agent] Gemini quota exhausted, switching to Ollama (Llama 3)...")
                            break
                    elif "API_KEY" in str(e) or "ADC" in str(e):
                        print(f"[Vision Agent] Gemini API key not configured, using Ollama (Llama 3)...")
                        break
                    raise e

            # Fallback to Ollama (Llama 3) - text-only analysis
            if ollama_client.is_available():
                print(f"[Vision Agent] Using Ollama (Llama 3) for threat analysis (text-only)...")
                ollama_response = ollama_client.generate(
                    prompt=prompt,
                    system="You are a humanitarian AI assistant specializing in conflict threat assessment from satellite imagery analysis.",
                    temperature=0.7,
                    max_tokens=300
                )
                if ollama_response:
                    return ollama_response

            # Final fallback if both fail
            return f"Analysis unavailable. {len(detections)} threats detected with threat score calculated."

        except Exception as e:
            print(f"[Vision Agent] LLM analysis error: {e}")
            return f"Analysis unavailable. {len(detections)} threats detected with threat score calculated."

    def _generate_embedding(self, image: np.ndarray, embedding_dim: int = 512) -> np.ndarray:
        """
        Generate vector embedding of image for similarity search.

        In production, use a pretrained vision model (ResNet, CLIP, etc.)
        For demo, use simple feature extraction.
        """
        # Resize image to standard size
        resized = cv2.resize(image, (224, 224)) if cv2 is not None else np.zeros((224, 224, 3))

        # Simple feature extraction: histogram + edge features
        # In production, replace with proper vision embeddings
        hist = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) if cv2 is not None else np.zeros(512)
        hist = hist.flatten()[:embedding_dim]

        # Normalize
        hist = hist / (np.linalg.norm(hist) + 1e-7)

        # Pad if necessary
        if len(hist) < embedding_dim:
            hist = np.pad(hist, (0, embedding_dim - len(hist)))

        return hist

    def detect_pattern_changes(self, location: str, recent_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect if threat patterns are changing at a location.

        Uses Vector Memory to compare current threats with historical patterns.
        This enables detection of military buildups or escalating conflicts.
        """
        recent_threat_scores = [a["threat_score"] for a in recent_analyses if "threat_score" in a]

        pattern_analysis = memory_manager.vector_memory.detect_pattern_change(
            location=location,
            recent_threat_scores=recent_threat_scores,
            window_size=10
        )

        return pattern_analysis


# Create global instance
vision_agent = VisionIntelligenceAgent()
