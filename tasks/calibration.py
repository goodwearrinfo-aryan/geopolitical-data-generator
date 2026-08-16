"""Bayesian calibration tasks for Celery.

Runs PyMC HMC/NUTS calibration asynchronously, exposing progress
via Celery task state for dashboard integration.
"""

from __future__ import annotations

from typing import Any, Dict

import requests


def get_api_base() -> str:
    """Get the API base URL from environment."""
    import os

    return os.getenv("VITE_API_BASE", "http://localhost:8000/api/v1")


def run_calibration() -> Dict[str, Any]:
    """Trigger Bayesian calibration run via API.

    Returns: Task ID and initial status
    """
    api_base = get_api_base()
    resp = requests.post(f"{api_base}/calibration/", timeout=30)
    if resp.status_code == 202:
        return {
            "task_id": resp.json().get("calibration_id", "unknown"),
            "status": "pending",
            "message": "Bayesian calibration started via API",
        }
    return {"status": "error", "message": f"HTTP {resp.status_code}"}


def get_calibration_results(cal_id: str) -> Dict[str, Any]:
    """Poll API for calibration results."""
    api_base = get_api_base()
    resp = requests.get(f"{api_base}/calibration/{cal_id}", timeout=30)
    if resp.status_code == 200:
        return {
            "status": resp.json().get("status", "unknown"),
            "parameters": resp.json().get("parameters", {}),
            "message": f"Calibration {cal_id}: {resp.json().get('calibration_id', cal_id)}",
        }
    return {"status": "error", "message": f"HTTP {resp.status_code}"}


def list_calibrations() -> Dict[str, Any]:
    """List calibration configurations."""
    api_base = get_api_base()
    resp = requests.get(f"{api_base}/calibration/", timeout=30)
    if resp.status_code == 200:
        return {
            "status": "success",
            "calibrations": resp.json().get("calibrations", []),
        }
    return {"status": "error", "message": f"HTTP {resp.status_code}"}
