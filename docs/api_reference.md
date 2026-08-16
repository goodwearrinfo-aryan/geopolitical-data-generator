# Geopolitical Data Generator API v0.1.0

*Generated from OpenAPI Specification*

### `GET` /api/v1/calibration
*List calibration configurations*
List available calibration configurations.
> **Response**: Successful Response

### `POST` /api/v1/calibration
*Run Bayesian calibration*
Run Bayesian calibration on fixture data.
> **Response**: OK

### `GET` /api/v1/calibration/{cal_id}
*Get calibration results*
Get results from a calibration run.
> **Response**: Successful Response

### `GET` /api/v1/exports
*List available export formats*
List available export formats.
> **Response**: Successful Response

### `GET` /api/v1/exports/{job_id}
*Download export for job*
Download export results for a job in the specified format.
> **Response**: Successful Response

### `GET` /api/v1/jobs
*List all jobs*
List all running jobs.
> **Response**: Successful Response

### `POST` /api/v1/jobs
*Create a new job*
Create a new job and run it in the background.
> **Response**: OK

### `GET` /api/v1/jobs/{job_id}
*Get job status*
Get the status of a specific job.
> **Response**: Successful Response

### `GET` /api/v1/scenarios
*List all scenario templates*
List available scenario templates.
> **Response**: Successful Response

### `POST` /api/v1/scenarios
*Create a new scenario*
Create a new scenario instance.
> **Response**: OK
