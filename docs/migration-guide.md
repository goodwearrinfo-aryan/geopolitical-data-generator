# Migration Guide: geopolitical-data-generator v1.0.0

## Upgrading from v0.x to v1.0.0

This guide outlines the breaking changes and migration steps required when upgrading
from version 0.x to version 1.0.0 of the Geopolitical Data Generator.

### 10.1 API Changes

#### FastAPI Endpoints

All API v1 endpoints have been restructured. The following changes apply:

| Old Path | New Path | Change |
|----------|----------|--------|
| `/scenarios` | `/api/v1/scenarios` | Added version prefix |
| `/jobs` | `/api/v1/jobs` | Added version prefix |
| `/exports` | `/api/v1/exports` | Added version prefix |
| `/calibration` | `/api/v1/calibration` | Added version prefix |

**Migration:** Update all API client calls to prepend `/api/v1/` to the previous paths.

#### Response Models

Response schemas have been formalized using Pydantic models. The following fields
have been added or changed:

- All endpoints now return standardized `status` and `request_id` fields
- Error responses follow RFC 7807 format (`type`, `title`, `status`, `detail`, `instance`)
- Calibration endpoints return posterior summaries as nested dicts instead of flat objects

**Migration:** Update deserialization code to handle the new response format.

### 10.2 Configuration Changes

#### pyproject.toml

The package now requires Python >=3.11 and has the following additional optional dependencies:

```toml
[project.optional-dependencies]
ray = ["ray>=2.0"]    # Distributed execution
sqlalchemy = ["sqlalchemy>=2.0"]  # SQL ORM support
pyyaml = ["pyyaml>=6.0"]  # YAML config support
```

**Migration:** Add the required dependencies to your environment.

#### Docker

The Dockerfile has been rebuilt on `python:3.11-slim` with pyarrow support. The
following environment variables are now required:

```bash
export CORS_ORIGINS="http://localhost:3000"
export RAY_ADDRESS="auto"  # or "local" for single-process mode
```

**Migration:** Update your Docker run commands to include the new environment variables.

### 10.3 Data Pipeline Changes

The WDI data fetch script now uses a different API endpoint structure. The
`scripts/fetch_wdi.py` script has been updated with the following changes:

- Changed from REST API to WB Python client
- Added mock data fallback for offline development
- Output schema: `wdi.parquet` now includes `stability_index` column

**Migration:** If you have existing `wdi.parquet` files from v0.x, run the
transformation script `scripts/transform.py --migrate-v0` to add the new column.

### 10.4 Bayesian Calibrator

The Bayesian calibration engine has been rewritten with proper PyMC model
context management. The following parameters have changed names:

| Old Name | New Name | Reason |
|----------|----------|--------|
| `coup_base_rate` | `coup_base_rate` | unchanged |
| `escalation_prob` | `escalation_lambda` | Geometric→Poisson likelihood |
| `gdp_shock_std` | `gdp_shock_std` | unchanged |
| `alliance_formation` | `alliance_formation` | unchanged |

**Migration:** If you have calibrated configs from v0.x, re-run calibration with
the new fixture data or manually update the parameter names in your JSON config.

### 10.5 Plugin System

The plugin entry-point system has been overhauled. Plugins must now declare
entry points in `pyproject.toml` under `[project.entry-points."geopolitical.exporters"]`:

```toml
[project.entry-points."geopolitical.exporters"]
my_exporter = "my_plugin:MyExporter"
```

The old `geopolitical.exports` entry point group has been deprecated.

**Migration:** Update your plugin's `pyproject.toml` to use the new entry point
group name.

### 10.6 Dashboard

The React dashboard URL has changed from `/dashboard` to `/ui`. The WebSocket
endpoint for progress streaming is now `ws://localhost:8000/api/v1/jobs/{job_id}/stream`.

**Migration:** Update your frontend application to use the new URL paths.

---

## Summary of Breaking Changes

| Category | Change |
|----------|--------|
| API | All v1 endpoints prefixed with `/api/v1/` |
| Config | New optional dependencies, env vars |
| Data | `wdi.parquet` schema includes `stability_index` |
| Calibration | Parameter names updated (escalation_prob→escalation_lambda) |
| Plugins | Entry point group renamed |
| Dashboard | UI path `/ui`, WebSocket endpoint changed |

---

## Automated Migration Script

A Python script is provided to assist with automated migration:

```bash
python scripts/migrate_v0_to_v1.py --input ./v0_data --output ./v1_data
```

This script will:
1. Migrate `wdi.parquet` files to include `stability_index`
2. Update calibration config JSON files to use new parameter names
3. Rewrite API client imports to use the new `/api/v1/` prefix
4. Update plugin entry points in `pyproject.toml`