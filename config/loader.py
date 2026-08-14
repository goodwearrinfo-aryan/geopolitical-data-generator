"""Configuration loader for geopolitical data generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
    custom_countries: list = None
    seed: int = 42
    output_dir: str = "./output"
    export_formats: list = None
    kafka_enabled: bool = False
    kafka_bootstrap: str = "localhost:9092"
    kafka_topic_prefix: str = "geopolitical"
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # Calibration
    calibration_mode: str = "hybrid"
    use_world_bank: bool = True
    use_inscr: bool = True
    use_acled: bool = True
    mcmc_draws: int = 1000
    mcmc_tune: int = 500
    mcmc_chains: int = 4
    backtest_years: int = 10
    validation_metrics: list = None
    
    # Domain configs
    political_enabled: bool = True
    conflict_enabled: bool = True
    diplomatic_enabled: bool = True
    economic_enabled: bool = True
    military_enabled: bool = True
    demographic_enabled: bool = True
    
    # Political
    regime_types: list = None
    election_cycle_years: int = 4
    coup_base_rate: float = 0.02
    protest_threshold: float = 0.6
    
    # Conflict
    escalation_levels: list = None
    conflict_types: list = None
    casualties_distribution: str = "pareto"
    
    # Diplomatic
    alliance_types: list = None
    sanction_types: list = None
    treaty_categories: list = None
    
    # Economic
    gdp_growth_model: str = "ces_energy"
    trade_gravity_enabled: bool = True
    resource_types: list = None
    aid_flow_model: str = "gravity"
    
    # Military
    expenditure_pct_gdp_range: list = None
    personnel_pct_pop_range: list = None
    equipment_categories: list = None
    
    # Demographic
    migration_model: str = "gravity_push_pull"
    urbanization_rate_range: list = None
    ethnicity_fragmentation_source: str = "fearon_laitin"
    
    # Ensemble
    ensemble_enabled: bool = True
    n_scenarios: int = 100
    sensitivity_method: str = "morris"
    n_morris_trajectories: int = 20
    n_sobol_samples: int = 1000
    vary_parameters: list = None
    
    # Weak signals
    weak_signals_enabled: bool = True
    weak_signal_methods: list = None
    weak_signal_window_years: int = 5
    weak_signal_threshold_sigma: float = 2.5

    def __post_init__(self):
        if self.custom_countries is None:
            self.custom_countries = []
        if self.export_formats is None:
            self.export_formats = ["parquet", "csv", "geojson"]
        if self.validation_metrics is None:
            self.validation_metrics = ["gdp_rmse", "conflict_auc", "regime_transition_acc"]
        if self.regime_types is None:
            self.regime_types = [r.value for r in RegimeType]
        if self.escalation_levels is None:
            self.escalation_levels = [c.value for c in ConflictIntensity]
        if self.conflict_types is None:
            self.conflict_types = [c.value for c in ConflictType]
        if self.alliance_types is None:
            self.alliance_types = [a.value for a in AllianceType]
        if self.sanction_types is None:
            self.sanction_types = [s.value for s in SanctionType]
        if self.treaty_categories is None:
            self.treaty_categories = [t.value for t in TreatyCategory]
        if self.resource_types is None:
            self.resource_types = [r.value for r in ResourceType]
        if self.expenditure_pct_gdp_range is None:
            self.expenditure_pct_gdp_range = [0.5, 5.0]
        if self.personnel_pct_pop_range is None:
            self.personnel_pct_pop_range = [0.1, 1.0]
        if self.equipment_categories is None:
            self.equipment_categories = ["land", "air", "naval", "cyber", "space", "nuclear"]
        if self.urbanization_rate_range is None:
            self.urbanization_rate_range = [0.5, 3.0]
        if self.vary_parameters is None:
            self.vary_parameters = [
                "domains.political.coup_base_rate",
                "domains.conflict.escalation_probability",
                "domains.economic.gdp_shock_std",
                "domains.diplomatic.alliance_formation_rate",
                "calibration.priors.regime_transition",
            ]
        if self.weak_signal_methods is None:
            self.weak_signal_methods = ["mahalanobis", "structural_break", "critical_slowing_down"]


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
        if isinstance(value, dict) and hasattr(config, attr_name.replace(".", "_")):
            nested_obj = getattr(config, attr_name.replace(".", "_"))
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
    result = {}
    for key in dir(config):
        if not key.startswith("_"):
            value = getattr(config, key)
            if not callable(value):
                result[key] = value
    return result