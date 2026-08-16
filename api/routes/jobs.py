"""Jobs routes - Celery-integrated.

Dispatches job creation to Celery workers.
Status polling via AsyncResult.
"""

from fastapi import APIRouter, status

router = APIRouter(tags=["jobs"])


@router.get("", summary="List all jobs", status_code=status.HTTP_200_OK)
async def list_jobs():
    """List all running jobs (placeholder until Celery result ready)."""
    return {"jobs": [], "note": "Use /jobs/{task_id} for Celery task status"}


@router.post(
    "", summary="Create a new job via Celery", status_code=status.HTTP_202_ACCEPTED
)
async def create_job():
    """Create a new job and dispatch to Celery worker."""
    from tasks.scenarios import run_scenario

    # Start scenario via Celery
    result = run_scenario.delay("ict")  # Could parameterize later

    return {
        "status": "started",
        "job_id": result["task_id"],
        "celery_task_id": result["task_id"],
        "message": "Job dispatched to Celery worker",
    }


@router.get(
    "/{task_id}", summary="Get job status via Celery", status_code=status.HTTP_200_OK
)
async def get_job_status(task_id: str):
    """Get the status of a Celery task."""
    from tasks.scenarios import get_task_status

    status = get_task_status(task_id)
    return {"job_id": task_id, **status}
