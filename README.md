# geopolitical-data-generator

**v1.0.0** — Causal geopolitical scenario simulation with Bayesian calibration,
distributed execution, and real-time dashboards.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production](https://img.shields.io/badge/status-production-brightgreen.svg)](https://github.com/aryanagarwal/geopolitical-data-generator)

## Overview

The **geopolitical-data-generator** is a Python platform for causal geopolitical
scenario simulation. It provides:

- **Scenario engine**: Deterministic & stochastic scenario generation with
  Bayesian parameter calibration
- **Distributed execution**: Ray backend + Local backend for parallel scenario
  running
- **6 export formats**: Arrow, Feather, Parquet, CSV, GeoJSON, Network
- **FastAPI server**: Full REST + WebSocket API for scenario execution, job
  management, and result downloads
- **Agent-based micro-foundations**: 5 agent types (Household, Firm, Government,
  Elite, Population) for bottom-up dynamic modeling
- **Plugin system**: Entry-point–based extensibility for custom exporters &
  policies
- **Dashboard**: React + WebSocket frontend with real-time progress streaming

## Quick Start

```bash
# Install
pip install -e .

# Run the CLI
python -m geopolitical_data_generator.cli --help

# Start the API server
uvicorn api.main:app --reload

# Access API at http://localhost:8000
# Access docs at http://localhost:8000/docs