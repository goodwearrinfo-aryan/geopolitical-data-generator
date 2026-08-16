"""Scenario execution tasks for Celery.

Runs geopolitical scenarios asynchronously via the FastAPI backend.
"""

from __future__ import annotations

from typing import Any, Dict

import requests


def get_api_base() -> str:
    """Get the API base URL from environment."""
    import os

    return os.getenv("VITE_API_BASE", "http://localhost:8000/api/v1")


def run_scenario(scenario_id: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a geopolitical scenario asynchronously.

    Args:
        scenario_id: Scenario key (ict, coup, economic, conflict)
        params: Scenario-specific parameters

    Returns: Task ID and initial status
    """
    api_base = get_api_base()
    payload = {"scenario_id": scenario_id}
    if params:
        payload.update(params)

    resp = requests.post(f"{api_base}/scenarios/", json=payload, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Failed to start scenario: {resp.status_code} {resp.text}")

    task_data = resp.json()
    return {
        "task_id": task_data.get("job_id", "unknown"),
        "scenario_id": scenario_id,
        "status": "pending",
        "message": "Scenario execution started via API",
    }


def list_scenarios() -> Dict[str, Any]:
    """List available scenario templates."""
    api_base = get_api_base()
    resp = requests.get(f"{api_base}/scenarios/", timeout=30)
    if resp.status_code == 200:
        return {"status": "success", "scenarios": resp.json().get("scenarios", [])}
    return {"status": "error", "message": f"HTTP {resp.status_code}"}


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Poll API for task/job status."""
    api_base = get_api_base()
    resp = requests.get(f"{api_base}/jobs/{task_id}", timeout=30)
    if resp.status_code == 200:
        return {
            "status": resp.json().get("status", "unknown"),
            "progress": resp.json().get("progress", 0),
        }
    return {"status": "error", "message": f"HTTP {resp.status_code}"}
