"""Scenarios routes - Celery-integrated."""

from fastapi import APIRouter, status

router = APIRouter(tags=["scenarios"])


@router.get("", summary="List scenario templates", status_code=status.HTTP_200_OK)
async def list_scenarios():
    """List available scenario templates via Celery task."""
    from tasks.scenarios import list_scenarios

    result = list_scenarios()
    if result["status"] == "success":
        return {"scenarios": result["scenarios"]}
    return {"scenarios": ["ict", "coup", "economic", "conflict"]}


@router.post(
    "", summary="Create a new scenario via Celery", status_code=status.HTTP_201_CREATED
)
async def create_scenario():
    """Create a new scenario via Celery."""
    from tasks.scenarios import run_scenario

    result = run_scenario.delay("ict")
    return {"status": "created", "scenario": "pending", "task_id": result["task_id"]}
