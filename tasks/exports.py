"""Export format tasks for Celery.

Handles 7 export formats (arrow, feather, parquet, csv, geojson, network, sqlite)
asynchronously via the FastAPI export endpoints.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import requests


def get_api_base() -> str:
    """Get the API base URL from environment."""
    return os.getenv("VITE_API_BASE", "http://localhost:8000/api/v1")


def run_export(job_id: str, format: str = "parquet") -> Dict[str, Any]:
    """Request export of a job in the specified format.

    Args:
        job_id: Celery task ID or job identifier
        format: Export format (arrow, feather, parquet, csv, geojson, network, sqlite)

    Returns: Task info with download URL when complete
    """
    api_base = get_api_base()
    valid_formats = [
        "arrow",
        "feather",
        "parquet",
        "csv",
        "geojson",
        "network",
        "sqlite",
    ]
    if format not in valid_formats:
        return {
            "status": "error",
            "message": f"Invalid format. Choose from: {valid_formats}",
        }

    resp = requests.get(
        f"{api_base}/exports/{job_id}", params={"format": format}, timeout=120
    )

    if resp.status_code == 200:
        # Export file ready for download
        content_type = resp.headers.get("content-type", "application/octet-stream")
        content_disposition = resp.headers.get("content-disposition", "")
        filename = (
            content_disposition.split("filename=")[1].strip('"')
            if "filename=" in content_disposition
            else f"results.{format}"
        )

        return {
            "status": "completed",
            "format": format,
            "download_url": f"data:{content_type};base64,{resp.text[:100]}...",  # truncated for safety
            "content_length": resp.headers.get("content-length", "unknown"),
            "filename": filename,
            "message": f"Export ready: results.{format}",
        }

    # Export not yet ready - polling info
    return {
        "status": "pending",
        "format": format,
        "message": f"Export in progress (HTTP {resp.status_code})",
        "progress": 0,
    }


def list_formats() -> Dict[str, Any]:
    """List available export formats from API."""
    api_base = get_api_base()
    resp = requests.get(f"{api_base}/exports/", timeout=30)
    if resp.status_code == 200:
        return {"status": "success", "formats": resp.json().get("formats", [])}
    return {"status": "error", "message": f"HTTP {resp.status_code}"}


def check_export_status(job_id: str, format: str = "parquet") -> Dict[str, Any]:
    """Check export task status."""
    api_base = get_api_base()
    resp = requests.get(f"{api_base}/jobs/{job_id}", timeout=30)
    if resp.status_code == 200:
        job_data = resp.json()
        return {
            "status": job_data.get("status", "unknown"),
            "progress": job_data.get("progress", 0),
            "error": job_data.get("error", None),
        }
    return {"status": "error", "message": f"HTTP {resp.status_code}"}
