#!/bin/bash
# Start Celery workers for distributed execution

# Scenarios worker (2 CPUs)
celery -A celery_app worker -Q scenarios -c 2 --prefetch-multiplier=1 --loglevel=info &

# Exports worker (4 CPUs)  
celery -A celery_app worker -Q exports -c 4 --prefetch-multiplier=1 --loglevel=info &

# Calibration worker (1 CPU, heavy computation)
celery -A celery_app worker -Q calibration -c 1 --prefetch-multiplier=1 --loglevel=info &

# General purpose worker (2 CPUs)
celery -A celery_app worker -Q general -c 2 --prefetch-multiplier=1 --loglevel=info &

echo "Workers started. Visit http://localhost:5555 for Flower monitoring"
