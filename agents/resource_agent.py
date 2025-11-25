"""
Resource Optimization Agent
Calculates optimal resource deployment based on displacement forecasts.
Implements sequential workflow: receives forecast → calculates needs → outputs to Communication Agent.
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np

import google.generativeai as genai

from config.config import ModelConfig, GEMINI_API_KEY
from utils.memory import memory_manager
from utils.ollama_client import ollama_client


class ResourceOptimizationAgent:
    """
    Agent 3: Resource Optimization
    Determines optimal aid infrastructure placement.

    Multi-Agent Feature: SEQUENTIAL workflow
    - Input: Forecast from Forecasting Agent
    - Output: Resource recommendations for Communication Agent
    """

    def __init__(self):
        """Initialize Resource Agent with Gemini for natural language recommendations."""
        self.agent_name = "resource_optimization"

        # Configure Gemini for natural language recommendations
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(ModelConfig.GEMINI_FLASH)

        # Resource calculation parameters
        self.resource_ratios = {
            "water_points": 250,  # 1 water point per 250 people
            "health_centers": 10000,  # 1 health center per 10,000 people
            "shelters": 5,  # 1 shelter unit per 5 people
            "food_distribution": 500,  # 1 distribution point per 500 people
            "sanitation": 50  # 1 latrine per 50 people
        }

    def calculate_resource_needs(self, forecast_data: Dict[str, Any],
                                 location: str,
                                 geographic_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate resource needs based on displacement forecast.

        Sequential Workflow:
        1. Receives forecast from Forecasting Agent
        2. Calculates resource requirements
        3. Generates recommendations for Communication Agent

        Args:
            forecast_data: Output from Forecasting Agent
            location: Geographic location (lat, lon)
            geographic_constraints: Terrain, existing infrastructure, etc.

        Returns:
            Dictionary containing:
                - resource_requirements: Quantities needed
                - deployment_timeline: When to deploy
                - optimal_locations: Recommended coordinates
                - priority_level: Resource priority
                - gemini_recommendations: Natural language guidance
        """
        start_time = datetime.now()

        try:
            # Extract forecast information
            predicted_displacement = forecast_data.get("predicted_displacement", 0)
            forecast_months = forecast_data.get("forecast_months", [])
            risk_factors = forecast_data.get("risk_factors", [])

            # Step 1: Calculate resource quantities
            resource_requirements = self._calculate_quantities(predicted_displacement)

            # Step 2: Determine deployment timeline
            deployment_timeline = self._plan_deployment_timeline(
                forecast_months=forecast_months,
                risk_factors=risk_factors
            )

            # Step 3: Calculate optimal locations (simplified for demo)
            optimal_locations = self._calculate_optimal_locations(
                location=location,
                expected_population=predicted_displacement,
                constraints=geographic_constraints or {}
            )

            # Step 4: Determine priority level
            priority_level = self._determine_priority(
                displacement_count=predicted_displacement,
                risk_factors=risk_factors
            )

            # Step 5: Generate Gemini natural language recommendations
            gemini_recommendations = self._gemini_generate_recommendations(
                location=location,
                predicted_displacement=predicted_displacement,
                resource_requirements=resource_requirements,
                deployment_timeline=deployment_timeline,
                priority_level=priority_level
            )

            # Prepare result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = {
                "resource_requirements": resource_requirements,
                "deployment_timeline": deployment_timeline,
                "optimal_locations": optimal_locations,
                "priority_level": priority_level,
                "gemini_recommendations": gemini_recommendations,
                "location": location,
                "predicted_displacement": predicted_displacement,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time
            }

            # Log to episodic memory
            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="calculate_resource_needs",
                input_data={"forecast_data": forecast_data, "location": location},
                output_data=result,
                status="success",
                metadata={"execution_time": execution_time}
            )

            return result

        except Exception as e:
            # Log error
            error_result = {
                "error": str(e),
                "location": location
            }

            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="calculate_resource_needs",
                input_data={"forecast_data": forecast_data, "location": location},
                output_data=error_result,
                status="error",
                metadata={"error_message": str(e)}
            )

            return error_result

    def _calculate_quantities(self, expected_population: int) -> Dict[str, int]:
        """Calculate required quantities of each resource type."""
        requirements = {}

        for resource_type, ratio in self.resource_ratios.items():
            required = int(np.ceil(expected_population / ratio))
            requirements[resource_type] = required

        return requirements

    def _plan_deployment_timeline(self, forecast_months: List[float],
                                  risk_factors: List[Dict]) -> List[Dict[str, Any]]:
        """
        Plan when to deploy each resource type.

        Phases:
        1. Immediate (0-1 month): Emergency supplies
        2. Short-term (1-3 months): Temporary infrastructure
        3. Medium-term (3-6 months): Semi-permanent facilities
        """
        timeline = []

        # Immediate deployment (if high-priority risk factors)
        high_priority_risks = [rf for rf in risk_factors if rf.get("weight", 0) > 0.3]
        if high_priority_risks:
            timeline.append({
                "phase": "immediate",
                "months": "0-1",
                "resources": ["food_distribution", "water_points"],
                "rationale": "High-priority threats detected"
            })

        # Short-term deployment
        if len(forecast_months) > 1 and forecast_months[1] > 100:
            timeline.append({
                "phase": "short_term",
                "months": "1-3",
                "resources": ["shelters", "sanitation", "health_centers"],
                "rationale": "Anticipated displacement within 3 months"
            })

        # Medium-term deployment
        if len(forecast_months) > 3:
            timeline.append({
                "phase": "medium_term",
                "months": "3-6",
                "resources": ["health_centers", "sanitation"],
                "rationale": "Sustained displacement expected"
            })

        return timeline

    def _calculate_optimal_locations(self, location: str, expected_population: int,
                                     constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate optimal locations for aid infrastructure.

        In production, use geographic optimization algorithms.
        For demo, generate sample locations near the conflict zone.
        """
        # Parse location (assuming "lat,lon" format)
        try:
            lat, lon = map(float, location.split(","))
        except:
            lat, lon = 0.0, 0.0

        # Generate sample optimal locations (simplified)
        locations = []

        # Water points (distributed)
        num_water_points = int(np.ceil(expected_population / self.resource_ratios["water_points"]))
        for i in range(min(num_water_points, 5)):  # Show max 5 for demo
            locations.append({
                "resource_type": "water_point",
                "coordinates": f"{lat + np.random.uniform(-0.1, 0.1):.4f},{lon + np.random.uniform(-0.1, 0.1):.4f}",
                "capacity": 250,
                "priority": "high"
            })

        # Health centers (centralized)
        num_health_centers = int(np.ceil(expected_population / self.resource_ratios["health_centers"]))
        for i in range(min(num_health_centers, 2)):
            locations.append({
                "resource_type": "health_center",
                "coordinates": f"{lat + np.random.uniform(-0.05, 0.05):.4f},{lon + np.random.uniform(-0.05, 0.05):.4f}",
                "capacity": 10000,
                "priority": "medium"
            })

        return locations

    def _determine_priority(self, displacement_count: int, risk_factors: List[Dict]) -> str:
        """Determine resource deployment priority level."""
        # Priority based on scale and risk
        if displacement_count > 10000 or any(rf.get("weight", 0) > 0.4 for rf in risk_factors):
            return "urgent"
        elif displacement_count > 5000 or any(rf.get("weight", 0) > 0.3 for rf in risk_factors):
            return "high"
        elif displacement_count > 1000:
            return "medium"
        else:
            return "low"

    def _gemini_generate_recommendations(self, location: str, predicted_displacement: int,
                                        resource_requirements: Dict[str, int],
                                        deployment_timeline: List[Dict],
                                        priority_level: str) -> str:
        """
        Use Gemini to generate natural language resource recommendations.

        Context Engineering: Specialized prompt for humanitarian logistics
        """
        # Format resource requirements
        resources_text = "\n".join([
            f"- {res.replace('_', ' ').title()}: {qty}"
            for res, qty in resource_requirements.items()
        ])

        # Format deployment timeline
        timeline_text = "\n".join([
            f"- {phase['phase'].replace('_', ' ').title()} ({phase['months']}): {', '.join(phase['resources'])}"
            for phase in deployment_timeline
        ]) if deployment_timeline else "Deploy all resources immediately"

        prompt = f"""You are a humanitarian logistics AI planning resource deployment for displaced populations.

Location: {location}
Predicted Displacement: {predicted_displacement:,} people
Priority Level: {priority_level.upper()}

Required Resources:
{resources_text}

Deployment Timeline:
{timeline_text}

Task: Provide concise operational recommendations (2-3 sentences) covering:
1. Immediate actions needed
2. Critical resource priorities
3. Logistics considerations (access, terrain, security)"""

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
                            print(f"[Resource Agent] Gemini quota limit hit, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"[Resource Agent] Gemini quota exhausted, switching to Ollama (Llama 3)...")
                            break
                    elif "API_KEY" in str(e) or "ADC" in str(e):
                        print(f"[Resource Agent] Gemini API key not configured, using Ollama (Llama 3)...")
                        break
                    raise e

            # Fallback to Ollama (Llama 3)
            if ollama_client.is_available():
                print(f"[Resource Agent] Using Ollama (Llama 3) for deployment recommendations...")
                ollama_response = ollama_client.generate(
                    prompt=prompt,
                    system="You are a humanitarian logistics AI assistant specializing in resource deployment planning.",
                    temperature=0.7,
                    max_tokens=300
                )
                if ollama_response:
                    return ollama_response

            # Final fallback if both fail
            return f"Deploy {predicted_displacement:,} person capacity across {len(resource_requirements)} resource types with {priority_level} priority."

        except Exception as e:
            print(f"[Resource Agent] LLM recommendations error: {e}")
            return f"Deploy {predicted_displacement:,} person capacity across {len(resource_requirements)} resource types with {priority_level} priority."


# Create global instance
resource_agent = ResourceOptimizationAgent()
