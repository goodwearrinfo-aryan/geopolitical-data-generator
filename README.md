# Geopolitical Data Generator

Enterprise-grade synthetic geopolitical data generator for world simulation, risk modeling, scenario analysis, and ML training.

## Features

- **Multi-domain simulation**: Political, conflict, diplomatic, economic, military, demographic
- **Calibrated parameters**: Bayesian/hybrid calibration against World Bank, INSCR, ACLED data
- **Ensemble runs**: Monte Carlo with sensitivity analysis (Morris, Sobol)
- **Multiple export formats**: Parquet, CSV, GeoJSON, Network (GraphML/JSON)
- **Streaming support**: Kafka, Neo4j integration
- **Weak signal detection**: Mahalanobis, structural breaks, critical slowing down
- **Reproducible**: Seed-based, config-driven, versioned outputs

## Quick Start

```bash
# Install
pip install -e .[dev]

# Generate default config
python -m geopolitical_data_generator init-config -o config.yaml

# Run single simulation
python -m geopolitical_data_generator run -o ./output --format parquet csv

# Run ensemble (100 scenarios)
python -m geopolitical_data_generator ensemble -n 100 -o ./output/ensemble --sensitivity

# Calibrate against historical data
python -m geopolitical_data_generator calibrate -d ./data --method hybrid
```

## Architecture

```
geopolitical-data-generator/
├── cli.py                      # Main CLI entry point
├── config/
│   ├── default.yaml            # Default configuration
│   └── loader.py               # Config loading/validation
├── schemas/
│   └── core.py                 # Pydantic data models
├── engine/
│   ├── temporal.py             # Time-stepping & event scheduling
│   ├── causal.py               # Causal propagation & conflict dynamics
│   ├── spatial.py              # Trade gravity, contagion, migration
│   └── ensemble.py             # Monte Carlo & sensitivity analysis
├── calibration/
│   └── engine.py               # Parameter calibration & validation
├── exporters/
│   └── __init__.py             # Parquet, CSV, GeoJSON, Network export
└── tests/
```

## Data Sources

| Source | Coverage | Use |
|--------|----------|-----|
| World Bank WDI | 1000+ indicators × 200+ countries × 60yr | Economic calibration |
| INSCR/CSP | Polity, coups, fragility, MEPV | Political validation |
| ACLED | Subnational conflict events | Conflict calibration |
| Natural Earth | Country boundaries (GeoJSON) | Spatial engine |

## Configuration

Key settings in `config.yaml`:

```yaml
simulation:
  start_year: 2020
  end_year: 2050
  timestep: "monthly"
  country_tier: "core"  # core (50), extended (195)

calibration:
  mode: "hybrid"  # bayesian, moment_matching, hybrid
  use_world_bank: true
  use_inscr: true
  use_acled: true

ensemble:
  enabled: true
  n_scenarios: 100
  sensitivity_method: "morris"
```

## Output Schema

### Countries (per timestep)
- ISO codes, regime type, stability, economy, military, demographics
- Resources, trade partners, diplomatic relations, sanctions

### Events
- Elections, coups, protests, regime changes, conflicts, treaties, sanctions
- Causal links with delay and strength

### Conflicts
- Type (interstate, civil, ethnic, terrorism, proxy)
- Intensity (tension → total), casualties, displacement, economic cost

### Networks
- Diplomatic relations (weighted graph)
- Trade flows (directed, valued)
- Alliances (defense, economic, cultural, intelligence)

## Sensitivity Analysis

```python
from engine.ensemble import EnsembleEngine

engine = EnsembleEngine(config)
results = engine.run_ensemble(100)
engine.run_morris_sensitivity(param_bounds, n_trajectories=20)
engine.run_sobol_sensitivity(param_bounds, n_samples=1000)
engine.export_results("./output")
```

## Calibration

```python
from calibration.engine import CalibrationEngine

cal = CalibrationEngine(config)
cal.load_world_bank_data("./data/world_bank")
cal.load_inscr_data("./data/inscr")
cal.load_acled_data("./data/acled/events.csv")

result = cal.calibrate(initial_state, method="hybrid")
metrics = cal.validate(result.optimal_values, initial_state)
```

## Requirements

- Python 3.10+
- NumPy, Pandas, PyArrow
- NetworkX, SciPy
- Faker (synthetic data)
- PyMC (optional, for Bayesian calibration)
- Confluent-Kafka, Neo4j (optional, for streaming)

## License

MIT