"""
Memory Management System for Multi-Agent Coordination.
Implements conversation memory, episodic memory, and vector memory.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np

from config.config import AgentConfig, LOGS_DIR


class ConversationMemory:
    """
    Stores historical predictions and actual outcomes for learning.
    Used by Forecasting Agent to improve accuracy over time.
    """

    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self.predictions: deque = deque(maxlen=max_items)
        self.actuals: deque = deque(maxlen=max_items)

    def add_prediction(self, timestamp: str, location: str, predicted_displacement: int,
                      confidence: float, metadata: Dict[str, Any]):
        """Store a displacement prediction."""
        entry = {
            "timestamp": timestamp,
            "location": location,
            "predicted_displacement": predicted_displacement,
            "confidence": confidence,
            "metadata": metadata
        }
        self.predictions.append(entry)

    def add_actual(self, timestamp: str, location: str, actual_displacement: int,
                   metadata: Dict[str, Any]):
        """Store actual displacement outcome for validation."""
        entry = {
            "timestamp": timestamp,
            "location": location,
            "actual_displacement": actual_displacement,
            "metadata": metadata
        }
        self.actuals.append(entry)

    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate prediction accuracy metrics (RMSE, MAE)."""
        if len(self.predictions) == 0 or len(self.actuals) == 0:
            return {"rmse": 0.0, "mae": 0.0, "samples": 0}

        # Match predictions with actuals by location and timestamp
        errors = []
        for pred in self.predictions:
            for actual in self.actuals:
                if (pred["location"] == actual["location"] and
                    pred["timestamp"] == actual["timestamp"]):
                    error = pred["predicted_displacement"] - actual["actual_displacement"]
                    errors.append(error)
                    break

        if not errors:
            return {"rmse": 0.0, "mae": 0.0, "samples": 0}

        errors = np.array(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "samples": len(errors)
        }

    def get_recent_predictions(self, n: int = 10) -> List[Dict]:
        """Retrieve the n most recent predictions."""
        return list(self.predictions)[-n:] if n <= len(self.predictions) else list(self.predictions)


class EpisodicMemory:
    """
    Logs all agent decisions and interactions for debugging.
    Used by Orchestrator to track multi-agent coordination.
    """

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or LOGS_DIR / f"episodic_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.episodes: List[Dict] = []
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Create log file if it doesn't exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.write_text("[]")

    def log_episode(self, agent_name: str, action: str, input_data: Dict[str, Any],
                    output_data: Dict[str, Any], status: str, metadata: Optional[Dict] = None):
        """
        Log an agent decision/action.

        Args:
            agent_name: Name of the agent (e.g., 'vision_intelligence')
            action: Action performed (e.g., 'detect_threats', 'forecast_displacement')
            input_data: Input provided to the agent
            output_data: Output produced by the agent
            status: Execution status ('success', 'error', 'warning')
            metadata: Additional metadata (execution time, error messages, etc.)
        """
        episode = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "action": action,
            "input_data": input_data,
            "output_data": output_data,
            "status": status,
            "metadata": metadata or {}
        }
        self.episodes.append(episode)

        # Persist to file
        self._save_to_file()

    def _save_to_file(self):
        """Save episodes to JSON file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.episodes, f, indent=2)
        except Exception as e:
            print(f"Error saving episodic memory: {e}")

    def get_episodes_by_agent(self, agent_name: str) -> List[Dict]:
        """Retrieve all episodes for a specific agent."""
        return [ep for ep in self.episodes if ep["agent_name"] == agent_name]

    def get_recent_episodes(self, n: int = 20) -> List[Dict]:
        """Retrieve the n most recent episodes."""
        return self.episodes[-n:] if n <= len(self.episodes) else self.episodes

    def get_error_episodes(self) -> List[Dict]:
        """Retrieve all episodes with errors."""
        return [ep for ep in self.episodes if ep["status"] == "error"]


class VectorMemory:
    """
    Stores vector representations of past satellite analyses.
    Used by Vision Agent to detect pattern changes (e.g., military buildup).
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []

    def add_embedding(self, embedding: np.ndarray, location: str, timestamp: str,
                     detections: List[str], threat_score: float):
        """
        Store a satellite image embedding with metadata.

        Args:
            embedding: Vector representation of the satellite image
            location: Geographic location (lat, lon)
            timestamp: When the image was captured
            detections: List of detected objects/threats
            threat_score: Overall threat score (0-10)
        """
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")

        self.embeddings.append(embedding)
        self.metadata.append({
            "location": location,
            "timestamp": timestamp,
            "detections": detections,
            "threat_score": threat_score
        })

    def find_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Find similar historical satellite analyses using cosine similarity.

        Args:
            query_embedding: Embedding to search for
            top_k: Number of similar results to return

        Returns:
            List of similar analyses with metadata and similarity scores
        """
        if len(self.embeddings) == 0:
            return []

        # Calculate cosine similarities
        similarities = []
        for idx, emb in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append({
                "index": idx,
                "similarity": float(similarity),
                "metadata": self.metadata[idx]
            })

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    def detect_pattern_change(self, location: str, recent_threat_scores: List[float],
                             window_size: int = 10) -> Dict[str, Any]:
        """
        Detect if threat patterns are changing at a location.

        Args:
            location: Geographic location to analyze
            recent_threat_scores: Recent threat scores for the location
            window_size: Number of historical samples to compare against

        Returns:
            Analysis of pattern change with trend direction and magnitude
        """
        # Filter metadata by location
        location_data = [
            (idx, meta) for idx, meta in enumerate(self.metadata)
            if meta["location"] == location
        ]

        if len(location_data) < window_size:
            return {
                "pattern_change_detected": False,
                "reason": "Insufficient historical data",
                "historical_avg": None,
                "recent_avg": None,
                "trend": None
            }

        # Get historical threat scores
        historical_scores = [meta["threat_score"] for _, meta in location_data[-window_size:]]
        historical_avg = np.mean(historical_scores)
        recent_avg = np.mean(recent_threat_scores)

        # Detect significant change (>20% difference)
        change_pct = ((recent_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
        pattern_change_detected = abs(change_pct) > 20

        return {
            "pattern_change_detected": pattern_change_detected,
            "historical_avg": float(historical_avg),
            "recent_avg": float(recent_avg),
            "change_percentage": float(change_pct),
            "trend": "increasing" if change_pct > 0 else "decreasing" if change_pct < 0 else "stable"
        }


class MemoryManager:
    """
    Central memory management system coordinating all memory types.
    Used by Orchestrator for comprehensive memory access.
    """

    def __init__(self):
        self.conversation_memory = ConversationMemory(max_items=AgentConfig.MAX_MEMORY_ITEMS)
        self.episodic_memory = EpisodicMemory()
        self.vector_memory = VectorMemory()

    def cleanup(self):
        """Perform memory cleanup when threshold is reached."""
        # Conversation memory uses deque with maxlen, so it auto-cleans
        # Episodic memory is persisted to file, so we only keep recent in memory
        if len(self.episodic_memory.episodes) > AgentConfig.MAX_MEMORY_ITEMS:
            # Keep only recent episodes in memory
            self.episodic_memory.episodes = self.episodic_memory.episodes[-AgentConfig.MAX_MEMORY_ITEMS:]

        # Vector memory cleanup: keep only recent embeddings if too many
        if len(self.vector_memory.embeddings) > AgentConfig.MAX_MEMORY_ITEMS:
            self.vector_memory.embeddings = self.vector_memory.embeddings[-AgentConfig.MAX_MEMORY_ITEMS:]
            self.vector_memory.metadata = self.vector_memory.metadata[-AgentConfig.MAX_MEMORY_ITEMS:]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all memory systems."""
        return {
            "conversation_memory": {
                "predictions": len(self.conversation_memory.predictions),
                "actuals": len(self.conversation_memory.actuals),
                "accuracy_metrics": self.conversation_memory.get_accuracy_metrics()
            },
            "episodic_memory": {
                "total_episodes": len(self.episodic_memory.episodes),
                "error_count": len(self.episodic_memory.get_error_episodes())
            },
            "vector_memory": {
                "embeddings_stored": len(self.vector_memory.embeddings)
            }
        }


# Global memory manager instance
memory_manager = MemoryManager()
