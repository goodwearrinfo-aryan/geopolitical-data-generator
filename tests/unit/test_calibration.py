"""Unit tests for calibration.engine module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import date
from unittest.mock import Mock, patch

from calibration.engine import CalibrationEngine, CalibrationTarget, CalibrationResult
from engine.temporal import TemporalEngine, TimestepFrequency
from schemas.core import SimulationState, Country, RegimeType, CountryTier


class TestCalibrationEngine:
    """Tests for CalibrationEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "calibration": {
                "mode": "hybrid",
                "mcmc_draws": 100,
                "mcmc_tune": 50,
                "mcmc_chains": 2,
            }
        }
        self.engine = CalibrationEngine(self.config)
    
    def test_initialization(self):
        """Test calibration engine initialization."""
        assert self.engine.config == self.config
        assert self.engine.calibration_config == self.config["calibration"]
        assert self.engine.world_bank_data == {}
        assert self.engine.inscr_data == {}
        assert self.engine.acled_data is None
        assert self.engine.param_bounds == {}
        assert self.engine.param_names == []
        assert self.engine.targets == []
    
    def test_define_parameters(self):
        """Test parameter definition."""
        bounds = self.engine.define_parameters()
        
        assert isinstance(bounds, dict)
        assert len(bounds) > 0
        assert self.engine.param_bounds == bounds
        assert self.engine.param_names == list(bounds.keys())
        
        # Check key parameters exist
        assert "coup_base_rate" in bounds
        assert "escalation_base_prob" in bounds
        assert "gdp_growth_trend" in bounds
        assert "trade_distance_elasticity" in bounds
        assert "alliance_formation_rate" in bounds
        
        # Check bounds are reasonable
        for name, (low, high) in bounds.items():
            assert low < high
            assert low >= 0
    
    def test_polity_to_regime_prob(self):
        """Test polity score to regime probability conversion."""
        # Democracy: polity 6-10
        prob = self.engine._polity_to_regime_prob(RegimeType.DEMOCRACY, 10)
        assert prob == 1.0
        
        prob = self.engine._polity_to_regime_prob(RegimeType.DEMOCRACY, 6)
        assert prob == 0.2
        
        prob = self.engine._polity_to_regime_prob(RegimeType.DEMOCRACY, 5)
        assert prob == 0.0
        
        # Autocracy: polity -10 to -6
        prob = self.engine._polity_to_regime_prob(RegimeType.AUTOCRACY, -10)
        assert prob == 1.0
        
        prob = self.engine._polity_to_regime_prob(RegimeType.AUTOCRACY, -6)
        assert prob == 0.2
        
        prob = self.engine._polity_to_regime_prob(RegimeType.AUTOCRACY, -5)
        assert prob == 0.0
        
        # Anocracy: polity -5 to 5
        prob = self.engine._polity_to_regime_prob(RegimeType.ANOCRACY, 0)
        assert prob == 1.0
        
        prob = self.engine._polity_to_regime_prob(RegimeType.ANOCRACY, 5)
        assert prob == 0.0
    
    def test_polity_to_regime(self):
        """Test polity score to regime type conversion."""
        assert self.engine._polity_to_regime(10) == RegimeType.DEMOCRACY
        assert self.engine._polity_to_regime(6) == RegimeType.DEMOCRACY
        assert self.engine._polity_to_regime(5) == RegimeType.ANOCRACY
        assert self.engine._polity_to_regime(0) == RegimeType.ANOCRACY
        assert self.engine._polity_to_regime(-5) == RegimeType.ANOCRACY
        assert self.engine._polity_to_regime(-6) == RegimeType.AUTOCRACY
        assert self.engine._polity_to_regime(-10) == RegimeType.AUTOCRACY
    
    def test_load_world_bank_data(self, tmp_path):
        """Test loading World Bank data."""
        # Create test CSV files
        wdi_dir = tmp_path / "world_bank"
        wdi_dir.mkdir()
        
        gdp_data = pd.DataFrame({
            "country": ["USA", "CHN"],
            "iso3": ["USA", "CHN"],
            "year": [2020, 2020],
            "gdp_usd": [21e12, 14e12],
        })
        gdp_data.to_csv(wdi_dir / "gdp.csv", index=False)
        
        pop_data = pd.DataFrame({
            "country": ["USA", "CHN"],
            "iso3": ["USA", "CHN"],
            "year": [2020, 2020],
            "population": [331_000_000, 1_412_000_000],
        })
        pop_data.to_csv(wdi_dir / "population.csv", index=False)
        
        # Load data
        self.engine.load_world_bank_data(str(wdi_dir))
        
        assert "gdp" in self.engine.world_bank_data
        assert "population" in self.engine.world_bank_data
        assert len(self.engine.world_bank_data["gdp"]) == 2
        assert len(self.engine.world_bank_data["population"]) == 2
    
    def test_load_inscr_data(self, tmp_path):
        """Test loading INSCR data."""
        inscr_dir = tmp_path / "inscr"
        inscr_dir.mkdir()
        
        polity_data = pd.DataFrame({
            "country": ["USA", "CHN"],
            "iso3": ["USA", "CHN"],
            "year": [2020, 2020],
            "polity2": [10, -7],
        })
        polity_data.to_csv(inscr_dir / "polity.csv", index=False)
        
        self.engine.load_inscr_data(str(inscr_dir))
        
        assert "polity" in self.engine.inscr_data
        assert len(self.engine.inscr_data["polity"]) == 2
    
    def test_load_acled_data(self, tmp_path):
        """Test loading ACLED data."""
        acled_dir = tmp_path / "acled"
        acled_dir.mkdir()
        
        events_data = pd.DataFrame({
            "event_id": [1, 2],
            "event_date": ["2020-01-01", "2020-01-02"],
            "year": [2020, 2020],
            "country": ["USA", "CHN"],
            "iso3": ["USA", "CHN"],
            "event_type": ["battle", "protests"],
            "fatalities": [10, 0],
        })
        events_data.to_csv(acled_dir / "events.csv", index=False)
        
        self.engine.load_acled_data(str(acled_dir / "events.csv"))
        
        assert self.engine.acled_data is not None
        assert len(self.engine.acled_data) == 2
    
    def test_create_targets_from_data(self):
        """Test creating calibration targets from loaded data."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        # Add test countries
        usa = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        chn = Country(
            iso3="CHN", iso2="CN", name="China",
            region="Asia", subregion="Eastern Asia",
            area_km2=9596961, population=1412000000,
            gdp_usd=18e12, gdp_per_capita_usd=12000,
            regime_type=RegimeType.AUTOCRACY,
        )
        state.countries["USA"] = usa
        state.countries["CHN"] = chn
        
        # Mock loaded data
        self.engine.world_bank_data = {
            "gdp": pd.DataFrame({
                "iso3": ["USA", "CHN"],
                "year": [2020, 2020],
                "gdp_usd": [21e12, 14e12],
            })
        }
        self.engine.inscr_data = {
            "polity": pd.DataFrame({
                "iso3": ["USA", "CHN"],
                "year": [2020, 2020],
                "polity2": [10, -7],
            })
        }
        
        targets = self.engine.create_targets_from_data(state)
        
        # Should create GDP targets for both countries
        gdp_targets = [t for t in targets if t.name.startswith("gdp_")]
        assert len(gdp_targets) == 2
        
        # Should create regime targets for both countries
        regime_targets = [t for t in targets if t.name.startswith("regime_")]
        assert len(regime_targets) == 2
    
    def test_objective_function(self):
        """Test objective function calculation."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        state.countries["USA"] = country
        
        # Create a target
        self.engine.targets = [
            CalibrationTarget(
                name="gdp_USA",
                observed_values=np.array([21e12]),
                predicted_values=np.array([25e12]),
                weight=1.0,
                transform="log",
            )
        ]
        self.engine.param_names = ["gdp_growth_trend"]
        
        # Test objective function
        params = np.array([0.025])
        error = self.engine.objective_function(params, state)
        
        assert isinstance(error, float)
        assert error >= 0
    
    def test_compute_target_moments(self):
        """Test computing target moments from data."""
        self.engine.world_bank_data = {
            "gdp": pd.DataFrame({
                "iso3": ["USA", "USA", "CHN", "CHN"],
                "year": [2019, 2020, 2019, 2020],
                "gdp_usd": [20e12, 21e12, 13e12, 14e12],
            })
        }
        self.engine.acled_data = pd.DataFrame({
            "year": [2019, 2020, 2019, 2020],
            "event_id": [1, 2, 3, 4],
        })
        self.engine.inscr_data = {
            "polity": pd.DataFrame({
                "country": ["USA", "USA", "CHN", "CHN"],
                "year": [2019, 2020, 2019, 2020],
                "polity2": [10, 10, -7, -7],
            })
        }
        
        moments = self.engine._compute_target_moments()
        
        assert len(moments) == 3
        assert all(isinstance(m, float) for m in moments)
    
    def test_simulate_moments(self):
        """Test simulating moments for given parameters."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        params = {
            "gdp_growth_trend": 0.025,
            "escalation_base_prob": 0.15,
            "regime_transition_prob": 0.01,
        }
        
        moments = self.engine._simulate_moments(params, state)
        
        assert len(moments) == 3
        assert all(isinstance(m, float) for m in moments)


class TestCalibrationTarget:
    """Tests for CalibrationTarget dataclass."""
    
    def test_creation(self):
        """Test creating a calibration target."""
        target = CalibrationTarget(
            name="test_target",
            observed_values=np.array([1.0, 2.0, 3.0]),
            predicted_values=np.array([1.1, 2.1, 3.1]),
            weight=2.0,
            transform="log",
        )
        
        assert target.name == "test_target"
        assert target.weight == 2.0
        assert target.transform == "log"
        assert len(target.observed_values) == 3


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""
    
    def test_creation(self):
        """Test creating a calibration result."""
        result = CalibrationResult(
            parameter_names=["param1", "param2"],
            optimal_values=np.array([0.5, 0.3]),
            objective_value=0.01,
            convergence=True,
            iterations=50,
            confidence_intervals=[(0.4, 0.6), (0.2, 0.4)],
        )
        
        assert result.parameter_names == ["param1", "param2"]
        assert np.array_equal(result.optimal_values, np.array([0.5, 0.3]))
        assert result.objective_value == 0.01
        assert result.convergence is True
        assert result.iterations == 50
        assert result.confidence_intervals == [(0.4, 0.6), (0.2, 0.4)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])