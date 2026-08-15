"""Configuration loader for geopolitical data generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from schemas.core import (
    CountryTier, RegimeType, ConflictType, ConflictIntensity,
    AllianceType, SanctionType, TreatyCategory, ResourceType, EventType
)


@dataclass
class SimulationConfig:
    start_year: int = 2020
    end_year: int = 2050
    timestep: str = "monthly"
    country_tier: CountryTier = CountryTier.CORE
    custom_countries: List[str] = field(default_factory=list)
    seed: int = 42
    output_dir: str = "./output"
    export_formats: List[str] = field(default_factory=lambda: ["parquet", "csv", "geojson"])
    kafka_enabled: bool = False
    kafka_bootstrap: str = "localhost:9092"
    kafka_topic_prefix: str = "geopolitical"
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


@dataclass
class CalibrationConfig:
    mode: str = "hybrid"
    use_world_bank: bool = True
    use_inscr: bool = True
    use_acled: bool = True
    mcmc_draws: int = 1000
    mcmc_tune: int = 500
    mcmc_chains: int = 4
    backtest_years: int = 10
    validation_metrics: List[str] = field(default_factory=lambda: ["gdp_rmse", "conflict_auc", "regime_transition_acc"])


@dataclass
class PoliticalDomainConfig:
    enabled: bool = True
    regime_types: List[str] = field(default_factory=lambda: [r.value for r in RegimeType])
    election_cycle_years: int = 4
    coup_base_rate: float = 0.02
    protest_threshold: float = 0.6


@dataclass
class ConflictDomainConfig:
    enabled: bool = True
    escalation_levels: List[str] = field(default_factory=lambda: [c.value for c in ConflictIntensity])
    conflict_types: List[str] = field(default_factory=lambda: [c.value for c in ConflictType])
    casualties_distribution: str = "pareto"


@dataclass
class DiplomaticDomainConfig:
    enabled: bool = True
    alliance_types: List[str] = field(default_factory=lambda: [a.value for a in AllianceType])
    sanction_types: List[str] = field(default_factory=lambda: [s.value for s in SanctionType])
    treaty_categories: List[str] = field(default_factory=lambda: [t.value for t in TreatyCategory])


@dataclass
class EconomicDomainConfig:
    enabled: bool = True
    gdp_growth_model: str = "ces_energy"
    trade_gravity_enabled: bool = True
    resource_types: List[str] = field(default_factory=lambda: [r.value for r in ResourceType])
    aid_flow_model: str = "gravity"


@dataclass
class MilitaryDomainConfig:
    enabled: bool = True
    expenditure_pct_gdp_range: List[float] = field(default_factory=lambda: [0.5, 5.0])
    personnel_pct_pop_range: List[float] = field(default_factory=lambda: [0.1, 1.0])
    equipment_categories: List[str] = field(default_factory=lambda: ["land", "air", "naval", "cyber", "space", "nuclear"])


@dataclass
class DemographicDomainConfig:
    enabled: bool = True
    migration_model: str = "gravity_push_pull"
    urbanization_rate_range: List[float] = field(default_factory=lambda: [0.5, 3.0])
    ethnicity_fragmentation_source: str = "fearon_laitin"


@dataclass
class DomainsConfig:
    political: PoliticalDomainConfig = field(default_factory=PoliticalDomainConfig)
    conflict: ConflictDomainConfig = field(default_factory=ConflictDomainConfig)
    diplomatic: DiplomaticDomainConfig = field(default_factory=DiplomaticDomainConfig)
    economic: EconomicDomainConfig = field(default_factory=EconomicDomainConfig)
    military: MilitaryDomainConfig = field(default_factory=MilitaryDomainConfig)
    demographic: DemographicDomainConfig = field(default_factory=DemographicDomainConfig)


@dataclass
class EnsembleConfig:
    enabled: bool = True
    n_scenarios: int = 100
    sensitivity_method: str = "morris"
    n_morris_trajectories: int = 20
    n_sobol_samples: int = 1000
    vary_parameters: List[str] = field(default_factory=lambda: [
        "domains.political.coup_base_rate",
        "domains.conflict.escalation_probability",
        "domains.economic.gdp_shock_std",
        "domains.diplomatic.alliance_formation_rate",
        "calibration.priors.regime_transition",
    ])


@dataclass
class WeakSignalsConfig:
    enabled: bool = True
    methods: List[str] = field(default_factory=lambda: ["mahalanobis", "structural_break", "critical_slowing_down"])
    window_years: int = 5
    threshold_sigma: float = 2.5


@dataclass
class SimulationConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    domains: DomainsConfig = field(default_factory=DomainsConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    weak_signals: WeakSignalsConfig = field(default_factory=WeakSignalsConfig)


def load_config(config_path: Optional[str] = None) -> SimulationConfig:
    """Load configuration from YAML file with defaults."""
    config = SimulationConfig()
    
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if data:
                _apply_dict_to_config(config, data)
    
    return config


def _apply_dict_to_config(config: SimulationConfig, data: Dict[str, Any], prefix: str = "") -> None:
    """Recursively apply dictionary values to config object."""
    for key, value in data.items():
        attr_name = f"{prefix}{key}" if prefix else key
        
        # Handle nested sections
        if isinstance(value, dict) and hasattr(config, attr_name):
            nested_obj = getattr(config, attr_name)
            if hasattr(nested_obj, "__dict__"):
                _apply_dict_to_config(nested_obj, value)
                continue
        
        # Direct attribute
        if hasattr(config, attr_name):
            setattr(config, attr_name, value)
        elif "." in attr_name:
            # Nested attribute like "domains.political.coup_base_rate"
            parts = attr_name.split(".")
            obj = config
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)


def save_config(config: SimulationConfig, output_path: str) -> None:
    """Save configuration to YAML file."""
    data = _config_to_dict(config)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Convert config object to dictionary."""
    from enum import Enum
    
    def _convert(val):
        if isinstance(val, Enum):
            return val.value
        elif hasattr(val, '__dict__'):
            # Dataclass or object with __dict__
            return {k: _convert(v) for k, v in val.__dict__.items() if not k.startswith("_")}
        elif isinstance(val, dict):
            return {k: _convert(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_convert(v) for v in val]
        return val
    
    result = {}
    for key in dir(config):
        if not key.startswith("_"):
            value = getattr(config, key)
            if not callable(value):
                result[key] = _convert(value)
    return result