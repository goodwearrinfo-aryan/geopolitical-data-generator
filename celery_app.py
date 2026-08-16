"""Celery application for geopolitical-data-generator.

Task queues:
  - scenarios: scenario execution (2 workers)
  - exports: export format downloads (4 workers)
  - calibration: Bayesian calibration (1 worker)
  - general: utility tasks (2 workers)

Usage:
  $ celery -A celery_app worker -Q scenarios -c 2 --prefetch-multiplier=1
  $ celery -A celery_app worker -Q exports -c 4 --prefetch-multiplier=1
  $ celery -A celery_app worker -Q calibration -c 1 --prefetch-multiplier=1
  $ celery -A celery_app flower --port=5555
"""

broker_url = "redis://localhost:6379/0"
result_backend = "redis://localhost:6379/1"

from celery import Celery

app = Celery(
    "geopolitical-data-generator",
    broker=broker_url,
    backend=result_backend,
    include=["tasks.scenarios", "tasks.exports", "tasks.calibration"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "tasks.scenarios.run_scenario": "scenarios",
        "tasks.scenarios.list_scenarios": "scenarios",
        "tasks.exports.run_export": "exports",
        "tasks.exports.list_formats": "exports",
        "tasks.calibration.run_calibration": "calibration",
        "tasks.calibration.get_results": "calibration",
    },
)

if __name__ == "__main__":
    app.start()
