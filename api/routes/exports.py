"""Exports routes - Celery-integrated."""

from fastapi import APIRouter, status

router = APIRouter(tags=["exports"])


@router.get("", summary="List export formats", status_code=status.HTTP_200_OK)
async def list_formats():
    """List available export formats via Celery task."""
    from tasks.exports import list_formats

    result = list_formats()
    if result["status"] == "success":
        return {"formats": result["formats"]}
    return {
        "formats": [
            "arrow",
            "feather",
            "parquet",
            "csv",
            "geojson",
            "network",
            "sqlite",
        ]
    }


@router.get(
    "/{job_id}", summary="Download export for job", status_code=status.HTTP_200_OK
)
async def download_export(job_id: str, format: str = "parquet"):
    """Download export results for a job in the specified format via Celery."""
    from tasks.exports import run_export

    result = run_export.delay(job_id, format)
    return {
        "job_id": job_id,
        "format": format,
        "task_id": result["task_id"],
        "status": result["status"],
        "message": result["message"],
    }
