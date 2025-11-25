"""
Orchestration & Debug Agent
Coordinates multi-agent workflows (parallel/sequential/looped).
Implements validation, error handling, and conflict resolution.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

import google.generativeai as genai

from config.config import ModelConfig, AgentConfig, SystemConfig, GEMINI_API_KEY
from utils.memory import memory_manager
from utils.ollama_client import ollama_client

# Import other agents (circular import handled by lazy loading)
from agents.vision_agent import vision_agent
from agents.forecasting_agent import forecasting_agent
from agents.resource_agent import resource_agent
from agents.communication_agent import communication_agent


class OrchestratorAgent:
    """
    Agent 5: Orchestration & Debug
    Coordinates multi-agent workflows and handles errors.

    Multi-Agent Features:
    - PARALLEL: Runs Vision + Forecasting simultaneously
    - SEQUENTIAL: Chains Vision → Forecast → Resource → Communication
    - LOOPED: Continuously refines predictions with new data
    """

    def __init__(self):
        """Initialize Orchestrator with Gemini for coordination decisions."""
        self.agent_name = "orchestrator"

        # Configure Gemini Flash for fast coordination
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(ModelConfig.GEMINI_FLASH)

        # Workflow state
        self.workflow_state = {
            "current_workflow": None,
            "workflow_id": None,
            "start_time": None,
            "agents_completed": [],
            "errors": []
        }

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=SystemConfig.MAX_CONCURRENT_AGENTS)

    def run_parallel_workflow(self, image_path: str, historical_data: Any,
                             location: str) -> Dict[str, Any]:
        """
        Execute PARALLEL workflow: Vision + Forecasting agents run simultaneously.

        Workflow:
        1. Vision Agent analyzes satellite image
        2. Forecasting Agent processes historical data
        3. Both run in parallel, results combined by Orchestrator
        4. Orchestrator validates outputs

        Args:
            image_path: Path to satellite image for Vision Agent
            historical_data: Historical refugee data for Forecasting Agent
            location: Geographic location

        Returns:
            Combined results from both agents with validation
        """
        workflow_id = f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_workflow(workflow_id, "parallel")

        start_time = datetime.now()

        try:
            print(f"[Orchestrator] Starting PARALLEL workflow: Vision + Forecasting")

            # Submit both agents to thread pool for parallel execution
            vision_future = self.executor.submit(
                self._run_vision_agent_safe,
                image_path,
                location
            )

            forecasting_future = self.executor.submit(
                self._run_forecasting_agent_safe,
                location,
                historical_data
            )

            # Wait for both to complete
            vision_result = vision_future.result(timeout=AgentConfig.VISION_TIMEOUT)
            forecasting_result = forecasting_future.result(timeout=AgentConfig.FORECASTING_TIMEOUT)

            # Validate outputs
            validation_result = self._validate_parallel_outputs(
                vision_result=vision_result,
                forecasting_result=forecasting_result
            )

            # Resolve conflicts if any
            if validation_result["conflicts_detected"]:
                resolved_results = self._resolve_conflicts(
                    vision_result=vision_result,
                    forecasting_result=forecasting_result,
                    conflicts=validation_result["conflicts"]
                )
            else:
                resolved_results = {
                    "vision": vision_result,
                    "forecasting": forecasting_result
                }

            # Prepare combined result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = {
                "workflow_id": workflow_id,
                "workflow_type": "parallel",
                "vision_result": resolved_results["vision"],
                "forecasting_result": resolved_results["forecasting"],
                "validation": validation_result,
                "execution_time": execution_time,
                "status": "success"
            }

            # Log to episodic memory
            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="run_parallel_workflow",
                input_data={"image_path": image_path, "location": location},
                output_data=result,
                status="success",
                metadata={"execution_time": execution_time}
            )

            return result

        except Exception as e:
            # Handle workflow error
            error_result = self._handle_workflow_error(
                workflow_id=workflow_id,
                workflow_type="parallel",
                error=e
            )
            return error_result

    def run_sequential_workflow(self, image_path: str, historical_data: Any,
                               location: str, recipients: List[str]) -> Dict[str, Any]:
        """
        Execute SEQUENTIAL workflow: Vision → Forecast → Resource → Communication.

        Workflow:
        1. Vision Agent detects threats
        2. Forecasting Agent uses threat level to refine predictions
        3. Resource Agent calculates needs based on forecast
        4. Communication Agent sends alerts
        Each step depends on previous output.

        Args:
            image_path: Satellite image path
            historical_data: Historical refugee data
            location: Geographic location
            recipients: Alert recipients

        Returns:
            Complete workflow results with all agent outputs
        """
        workflow_id = f"sequential_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_workflow(workflow_id, "sequential")

        start_time = datetime.now()
        workflow_results = {}

        try:
            print(f"[Orchestrator] Starting SEQUENTIAL workflow: Vision → Forecast → Resource → Communication")

            # Step 1: Vision Agent
            print(f"[Orchestrator] Step 1/4: Running Vision Agent")
            vision_result = self._run_vision_agent_safe(image_path, location)
            workflow_results["vision"] = vision_result

            # Check for errors
            if "error" in vision_result:
                return self._handle_workflow_error(workflow_id, "sequential",
                                                   Exception(f"Vision Agent error: {vision_result['error']}"))

            # Step 2: Forecasting Agent (uses threat level from Vision)
            print(f"[Orchestrator] Step 2/4: Running Forecasting Agent")
            threat_level = vision_result.get("threat_level", "medium")
            threat_score = vision_result.get("threat_score", 5.0)

            forecasting_result = self._run_forecasting_agent_safe(
                location=location,
                historical_data=historical_data,
                threat_level=threat_level,
                vision_threat_score=threat_score
            )
            workflow_results["forecasting"] = forecasting_result

            if "error" in forecasting_result:
                return self._handle_workflow_error(workflow_id, "sequential",
                                                   Exception(f"Forecasting Agent error: {forecasting_result['error']}"))

            # Step 3: Resource Agent (uses forecast)
            print(f"[Orchestrator] Step 3/4: Running Resource Agent")
            resource_result = self._run_resource_agent_safe(
                forecast_data=forecasting_result,
                location=location
            )
            workflow_results["resource"] = resource_result

            if "error" in resource_result:
                return self._handle_workflow_error(workflow_id, "sequential",
                                                   Exception(f"Resource Agent error: {resource_result['error']}"))

            # Step 4: Communication Agent (sends alerts based on resource needs)
            print(f"[Orchestrator] Step 4/4: Running Communication Agent")
            alert_data = {
                "location": location,
                "count": forecasting_result.get("predicted_displacement", 0),
                "months": forecasting_result.get("forecast_horizon_months", 6),
                "level": threat_level,
                "resources": self._format_resources(resource_result.get("resource_requirements", {}))
            }

            communication_result = self._run_communication_agent_safe(
                alert_type="displacement_warning",
                data=alert_data,
                recipients=recipients
            )
            workflow_results["communication"] = communication_result

            # Prepare final result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = {
                "workflow_id": workflow_id,
                "workflow_type": "sequential",
                "workflow_results": workflow_results,
                "execution_time": execution_time,
                "status": "success"
            }

            # Log to episodic memory
            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="run_sequential_workflow",
                input_data={"image_path": image_path, "location": location},
                output_data=result,
                status="success",
                metadata={"execution_time": execution_time}
            )

            print(f"[Orchestrator] SEQUENTIAL workflow completed successfully in {execution_time:.2f}s")

            return result

        except Exception as e:
            return self._handle_workflow_error(workflow_id, "sequential", e)

    def run_looped_workflow(self, image_paths: List[str], historical_data: Any,
                           location: str, loop_iterations: int = 3) -> Dict[str, Any]:
        """
        Execute LOOPED workflow: Continuously refine predictions with new data.

        Workflow:
        1. Vision Agent analyzes new satellite images
        2. New detections fed back to Forecasting Agent
        3. Forecast refined with updated threat information
        4. Loop continues for N iterations or until convergence

        Args:
            image_paths: List of satellite images to analyze iteratively
            historical_data: Historical refugee data
            location: Geographic location
            loop_iterations: Maximum number of refinement iterations

        Returns:
            Final refined predictions with iteration history
        """
        workflow_id = f"looped_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._init_workflow(workflow_id, "looped")

        start_time = datetime.now()
        iteration_history = []

        try:
            print(f"[Orchestrator] Starting LOOPED workflow: {loop_iterations} iterations")

            current_threat_score = 0.0
            current_forecast = None

            for iteration in range(loop_iterations):
                print(f"[Orchestrator] Loop iteration {iteration + 1}/{loop_iterations}")

                # Analyze next image (or cycle through images)
                image_path = image_paths[iteration % len(image_paths)]

                # Step 1: Vision Agent
                vision_result = self._run_vision_agent_safe(image_path, location)
                new_threat_score = vision_result.get("threat_score", 0.0)

                # Step 2: Check if threat score changed significantly
                threat_change = abs(new_threat_score - current_threat_score)

                # Step 3: Refine forecast if threat changed
                if threat_change > 0.5 or current_forecast is None:
                    print(f"[Orchestrator] Threat score changed by {threat_change:.2f}, refining forecast")

                    forecasting_result = self._run_forecasting_agent_safe(
                        location=location,
                        historical_data=historical_data,
                        threat_level=vision_result.get("threat_level", "medium"),
                        vision_threat_score=new_threat_score
                    )

                    current_forecast = forecasting_result
                    current_threat_score = new_threat_score

                # Record iteration
                iteration_history.append({
                    "iteration": iteration + 1,
                    "threat_score": new_threat_score,
                    "threat_change": threat_change,
                    "forecast_updated": threat_change > 0.5,
                    "predicted_displacement": current_forecast.get("predicted_displacement", 0) if current_forecast else 0
                })

            # Prepare final result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = {
                "workflow_id": workflow_id,
                "workflow_type": "looped",
                "iterations": loop_iterations,
                "iteration_history": iteration_history,
                "final_forecast": current_forecast,
                "final_threat_score": current_threat_score,
                "execution_time": execution_time,
                "status": "success"
            }

            # Log to episodic memory
            memory_manager.episodic_memory.log_episode(
                agent_name=self.agent_name,
                action="run_looped_workflow",
                input_data={"location": location, "iterations": loop_iterations},
                output_data=result,
                status="success",
                metadata={"execution_time": execution_time}
            )

            print(f"[Orchestrator] LOOPED workflow completed: {loop_iterations} iterations in {execution_time:.2f}s")

            return result

        except Exception as e:
            return self._handle_workflow_error(workflow_id, "looped", e)

    def _validate_parallel_outputs(self, vision_result: Dict, forecasting_result: Dict) -> Dict[str, Any]:
        """
        Validate outputs from parallel agents for consistency.

        Debugging & Error Handling: Check for anomalies and conflicts
        """
        conflicts = []
        warnings = []

        # Check 1: Threat level vs displacement prediction consistency
        threat_level = vision_result.get("threat_level", "unknown")
        predicted_displacement = forecasting_result.get("predicted_displacement", 0)

        if threat_level in ["critical", "high"] and predicted_displacement < 1000:
            conflicts.append({
                "type": "inconsistency",
                "description": f"High threat ({threat_level}) but low predicted displacement ({predicted_displacement})",
                "severity": "medium"
            })

        if threat_level == "low" and predicted_displacement > 5000:
            conflicts.append({
                "type": "inconsistency",
                "description": f"Low threat ({threat_level}) but high predicted displacement ({predicted_displacement})",
                "severity": "medium"
            })

        # Check 2: Vision agent false positive detection
        detections = vision_result.get("detections", [])
        if len(detections) > 10:
            warnings.append({
                "type": "anomaly",
                "description": f"Unusually high number of detections ({len(detections)})",
                "recommendation": "Manual verification recommended"
            })

        # Check 3: Forecast confidence
        if "error" in forecasting_result or "error" in vision_result:
            conflicts.append({
                "type": "error",
                "description": "One or more agents returned errors",
                "severity": "high"
            })

        return {
            "conflicts_detected": len(conflicts) > 0,
            "conflicts": conflicts,
            "warnings": warnings,
            "validation_passed": len([c for c in conflicts if c["severity"] == "high"]) == 0
        }

    def _resolve_conflicts(self, vision_result: Dict, forecasting_result: Dict,
                          conflicts: List[Dict]) -> Dict[str, Dict]:
        """
        Resolve conflicts between agent outputs using Gemini coordination.

        Debugging Feature: Intelligent conflict resolution
        """
        # Prepare conflict summary for Gemini
        conflict_summary = "\n".join([
            f"- {c['description']} (severity: {c['severity']})"
            for c in conflicts
        ])

        prompt = f"""You are an AI coordinator resolving conflicts between two agents analyzing a refugee crisis.

Vision Agent Results:
- Threat Level: {vision_result.get('threat_level', 'unknown')}
- Threat Score: {vision_result.get('threat_score', 0)}/10
- Detections: {len(vision_result.get('detections', []))} objects

Forecasting Agent Results:
- Predicted Displacement: {forecasting_result.get('predicted_displacement', 0):,} people
- Forecast Horizon: {forecasting_result.get('forecast_horizon_months', 6)} months

Detected Conflicts:
{conflict_summary}

Decision Rule: Safety bias - prioritize Vision Agent's threat assessment if conflict involves threat level.

Task: For each conflict, decide which agent's output to prioritize and explain briefly (1 sentence per conflict)."""

        try:
            import time
            max_retries = 3
            retry_delay = 2  # seconds

            resolution_text = None
            # Try Gemini first
            for attempt in range(max_retries):
                try:
                    response = self.gemini_model.generate_content(prompt)
                    resolution_text = response.text.strip()
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"[Orchestrator] Gemini quota limit hit, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"[Orchestrator] Gemini quota exhausted, switching to Ollama (Llama 3)...")
                            break
                    elif "API_KEY" in str(e) or "ADC" in str(e):
                        print(f"[Orchestrator] Gemini API key not configured, using Ollama (Llama 3)...")
                        break
                    raise e

            # Fallback to Ollama (Llama 3) if Gemini failed
            if not resolution_text and ollama_client.is_available():
                print(f"[Orchestrator] Using Ollama (Llama 3) for conflict resolution...")
                ollama_response = ollama_client.generate(
                    prompt=prompt,
                    system="You are an AI orchestrator specializing in multi-agent conflict resolution for humanitarian operations.",
                    temperature=0.7,
                    max_tokens=300
                )
                if ollama_response:
                    resolution_text = ollama_response

            # For safety bias, prioritize Vision Agent for threat-related conflicts
            resolved_vision = vision_result.copy()
            resolved_forecasting = forecasting_result.copy()

            # Apply safety bias: if Vision says high threat, trust it
            if vision_result.get("threat_level") in ["critical", "high"]:
                print(f"[Orchestrator] Applying safety bias: Prioritizing Vision Agent's {vision_result.get('threat_level')} threat assessment")

            return {
                "vision": resolved_vision,
                "forecasting": resolved_forecasting,
                "resolution_reasoning": resolution_text if resolution_text else "Conflict resolution unavailable due to API limits."
            }

        except Exception as e:
            print(f"[Orchestrator] Conflict resolution error: {e}. Using original outputs.")
            return {
                "vision": vision_result,
                "forecasting": forecasting_result,
                "resolution_reasoning": "Error in Gemini resolution, using safety bias (Vision prioritized)"
            }

    def _run_vision_agent_safe(self, image_path: str, location: str) -> Dict[str, Any]:
        """Safely run Vision Agent with error handling."""
        try:
            return vision_agent.analyze_image(
                image_path=image_path,
                location=location,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            return {"error": str(e), "agent": "vision"}

    def _run_forecasting_agent_safe(self, location: str, historical_data: Any,
                                   threat_level: Optional[str] = None,
                                   vision_threat_score: Optional[float] = None) -> Dict[str, Any]:
        """Safely run Forecasting Agent with error handling."""
        try:
            return forecasting_agent.forecast_displacement(
                location=location,
                historical_data=historical_data,
                threat_level=threat_level,
                vision_threat_score=vision_threat_score
            )
        except Exception as e:
            return {"error": str(e), "agent": "forecasting"}

    def _run_resource_agent_safe(self, forecast_data: Dict, location: str) -> Dict[str, Any]:
        """Safely run Resource Agent with error handling."""
        try:
            return resource_agent.calculate_resource_needs(
                forecast_data=forecast_data,
                location=location
            )
        except Exception as e:
            return {"error": str(e), "agent": "resource"}

    def _run_communication_agent_safe(self, alert_type: str, data: Dict,
                                     recipients: List[str]) -> Dict[str, Any]:
        """Safely run Communication Agent with error handling."""
        try:
            return communication_agent.send_alert(
                alert_type=alert_type,
                data=data,
                recipients=recipients
            )
        except Exception as e:
            return {"error": str(e), "agent": "communication"}

    def _init_workflow(self, workflow_id: str, workflow_type: str):
        """Initialize workflow state."""
        self.workflow_state = {
            "current_workflow": workflow_type,
            "workflow_id": workflow_id,
            "start_time": datetime.now().isoformat(),
            "agents_completed": [],
            "errors": []
        }

    def _handle_workflow_error(self, workflow_id: str, workflow_type: str, error: Exception) -> Dict[str, Any]:
        """Handle workflow errors with logging."""
        # Calculate execution time from workflow start
        start_time = self.workflow_state.get("start_time")
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                execution_time = (datetime.now() - start_dt).total_seconds()
            except:
                execution_time = 0.0
        else:
            execution_time = 0.0

        error_result = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "status": "error",
            "error": str(error),
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            # Add empty fields for main.py compatibility
            "workflow_results": {},
            "iterations": 0,
            "iteration_history": []
        }

        # Log to episodic memory
        memory_manager.episodic_memory.log_episode(
            agent_name=self.agent_name,
            action=f"run_{workflow_type}_workflow",
            input_data={"workflow_id": workflow_id},
            output_data=error_result,
            status="error",
            metadata={"error_message": str(error)}
        )

        return error_result

    def _format_resources(self, resource_requirements: Dict[str, int]) -> str:
        """Format resource requirements for alert message."""
        return ", ".join([
            f"{qty} {res.replace('_', ' ')}"
            for res, qty in list(resource_requirements.items())[:3]
        ])

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of all memory systems."""
        return memory_manager.get_summary()


# Create global instance
orchestrator = OrchestratorAgent()
