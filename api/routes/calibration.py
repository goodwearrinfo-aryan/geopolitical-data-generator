"""Calibration routes - Celery-integrated."""

from fastapi import APIRouter, status

router = APIRouter(tags=["calibration"])


@router.get(
    "", summary="List calibration configurations", status_code=status.HTTP_200_OK
)
async def list_calibrations():
    """List calibration configurations via Celery task."""
    from tasks.calibration import list_calibrations

    result = list_calibrations()
    if result["status"] == "success":
        return {"calibrations": result["calibrations"]}
    return {"calibrations": []}


@router.post(
    "",
    summary="Run Bayesian calibration via Celery",
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_calibration():
    """Run Bayesian calibration via Celery worker."""
    from tasks.calibration import run_calibration

    result = run_calibration.delay()
    return {
        "status": "started",
        "calibration_id": result["task_id"],
        "celery_task_id": result["task_id"],
        "message": "Calibration dispatched to Celery worker",
    }


@router.get(
    "/{cal_id}", summary="Get calibration results", status_code=status.HTTP_200_OK
)
async def get_calibration_results(cal_id: str):
    """Get results from a Celery calibration run."""
    from tasks.calibration import get_calibration_results

    result = get_calibration_results(cal_id)
    return result
