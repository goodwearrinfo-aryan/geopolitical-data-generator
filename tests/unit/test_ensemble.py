"""Unit tests for engine.ensemble module."""

from __future__ import annotations

import pytest
import numpy as np

from engine.ensemble import EnsembleEngine, EnsembleResult, SensitivityResult
from engine.temporal import TemporalEngine, TimestepFrequency


class TestEnsembleEngine:
    """Tests for EnsembleEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "simulation": {"seed": 42},
            "ensemble": {"n_workers": 2},
        }
        self.engine = EnsembleEngine(self.config, n_workers=2, seed=42)
    
    def test_initialization(self):
        """Test ensemble engine initialization."""
        assert self.engine.base_config == self.config
        assert self.engine.n_workers == 2
        assert self.engine.base_seed == 42
        assert self.engine.rng is not None
        assert self.engine.results == []
        assert self.engine.sensitivity_results == []
    
    def test_get_default_variations(self):
        """Test default parameter variations."""
        variations = self.engine._get_default_variations()
        
        assert isinstance(variations, dict)
        assert len(variations) == 5
        assert "domains.political.coup_base_rate" in variations
        assert "domains.conflict.escalation_probability" in variations
        assert "domains.economic.gdp_shock_std" in variations
        assert "domains.diplomatic.alliance_formation_rate" in variations
        assert "calibration.priors.regime_transition" in variations
        
        # Check bounds are tuples
        for param, bounds in variations.items():
            assert isinstance(bounds, tuple)
            assert len(bounds) == 2
            assert bounds[0] < bounds[1]
    
    def test_generate_parameter_sets_lhs(self):
        """Test Latin Hypercube Sampling parameter generation."""
        variations = {
            "param1": (0.0, 1.0),
            "param2": (10.0, 20.0),
            "param3": (-1.0, 1.0),
        }
        
        n_sets = 100
        param_sets = self.engine._generate_parameter_sets(variations, n_sets)
        
        assert len(param_sets) == n_sets
        assert all(len(ps) == 3 for ps in param_sets)
        assert all("param1" in ps for ps in param_sets)
        assert all("param2" in ps for ps in param_sets)
        assert all("param3" in ps for ps in param_sets)
        
        # Check bounds are respected
        for ps in param_sets:
            assert 0.0 <= ps["param1"] <= 1.0
            assert 10.0 <= ps["param2"] <= 20.0
            assert -1.0 <= ps["param3"] <= 1.0
        
        # Check LHS property: each parameter should have good coverage
        for param, (low, high) in variations.items():
            values = sorted([ps[param] for ps in param_sets])
            # Should span the range
            assert values[0] <= low + (high - low) * 0.1  # Near lower bound
            assert values[-1] >= high - (high - low) * 0.1  # Near upper bound
    
    def test_run_single_scenario(self):
        """Test running a single scenario."""
        params = {
            "domains.political.coup_base_rate": 0.03,
            "domains.conflict.escalation_probability": 0.2,
        }
        
        result = self.engine._run_single_scenario(0, 12345, params)
        
        assert isinstance(result, EnsembleResult)
        assert result.run_id == 0
        assert result.seed == 12345
        assert isinstance(result.metrics, dict)
        assert "final_global_gdp" in result.metrics
        assert "avg_growth_rate" in result.metrics
        assert "peak_conflicts" in result.metrics
        assert len(result.global_gdp_trajectory) == 360  # 30 years * 12 months
        assert len(result.global_conflict_trajectory) == 360
    
    def test_run_ensemble(self):
        """Test running ensemble with multiple scenarios."""
        n_scenarios = 5
        param_variations = {
            "param1": (0.0, 1.0),
        }
        
        results = self.engine.run_ensemble(n_scenarios, param_variations)
        
        assert len(results) == n_scenarios
        assert all(isinstance(r, EnsembleResult) for r in results)
        assert len(set(r.run_id for r in results)) == n_scenarios
        assert len(set(r.seed for r in results)) == n_scenarios
        
        # Results should be sorted by run_id
        assert results[0].run_id == 0
        assert results[-1].run_id == n_scenarios - 1
    
    def test_run_morris_sensitivity(self):
        """Test Morris sensitivity analysis."""
        param_bounds = {
            "param1": (0.0, 1.0),
            "param2": (0.0, 1.0),
            "param3": (0.0, 1.0),
        }
        
        results = self.engine.run_morris_sensitivity(param_bounds, n_trajectories=4, n_steps=2)
        
        assert len(results) == 3
        assert all(isinstance(r, SensitivityResult) for r in results)
        assert all(r.method == "morris" for r in results)
        assert all(r.mu_star >= 0 for r in results)
        assert all(r.sigma >= 0 for r in results)
        assert all(len(r.confidence_interval) == 2 for r in results)
        
        # Results should be stored in engine
        assert len(self.engine.sensitivity_results) == 3
    
    def test_run_sobol_sensitivity(self):
        """Test Sobol sensitivity analysis."""
        param_bounds = {
            "param1": (0.0, 1.0),
            "param2": (0.0, 1.0),
        }
        
        results = self.engine.run_sobol_sensitivity(param_bounds, n_samples=100)
        
        assert len(results) == 2
        assert all(isinstance(r, SensitivityResult) for r in results)
        assert all(r.method == "sobol" for r in results)
        assert all(0.0 <= r.sobol_first <= 1.0 for r in results)
        assert all(0.0 <= r.sobol_total <= 1.0 for r in results)
        assert all(r.sobol_first <= r.sobol_total for r in results)
        
        # Results should be stored
        assert len(self.engine.sensitivity_results) == 2
    
    def test_get_summary_statistics(self):
        """Test summary statistics calculation."""
        # Run a small ensemble
        self.engine.run_ensemble(3, {"param1": (0.0, 1.0)})
        
        summary = self.engine.get_summary_statistics()
        
        assert "final_global_gdp" in summary
        assert "mean" in summary["final_global_gdp"]
        assert "std" in summary["final_global_gdp"]
        assert "min" in summary["final_global_gdp"]
        assert "max" in summary["final_global_gdp"]
        assert "median" in summary["final_global_gdp"]
        
        # Trajectory statistics
        assert "gdp_trajectory" in summary
        assert "mean" in summary["gdp_trajectory"]
        assert "std" in summary["gdp_trajectory"]
        assert "q25" in summary["gdp_trajectory"]
        assert "q75" in summary["gdp_trajectory"]
        assert len(summary["gdp_trajectory"]["mean"]) == 360
        
        assert "conflict_trajectory" in summary
        assert len(summary["conflict_trajectory"]["mean"]) == 360
    
    def test_empty_summary(self):
        """Test summary with no results."""
        summary = self.engine.get_summary_statistics()
        assert summary == {}


class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""
    
    def test_creation(self):
        """Test creating an EnsembleResult."""
        result = EnsembleResult(
            run_id=0,
            seed=12345,
            final_state=None,
            metrics={"gdp": 1e12, "conflicts": 5},
            trajectory=[{"step": 0, "gdp": 1e12}],
            global_gdp_trajectory=[1e12, 1.01e12],
            global_conflict_trajectory=[5, 4],
            regime_changes=2,
            conflicts_started=1,
            wars=0,
            nuclear_events=0,
        )
        
        assert result.run_id == 0
        assert result.seed == 12345
        assert result.metrics["gdp"] == 1e12
        assert len(result.global_gdp_trajectory) == 2


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""
    
    def test_morris_result(self):
        """Test Morris sensitivity result."""
        result = SensitivityResult(
            parameter="param1",
            method="morris",
            mu_star=0.5,
            sigma=0.1,
            confidence_interval=(0.4, 0.6),
        )
        
        assert result.parameter == "param1"
        assert result.method == "morris"
        assert result.mu_star == 0.5
        assert result.sigma == 0.1
        assert result.confidence_interval == (0.4, 0.6)
        assert result.sobol_first is None
        assert result.sobol_total is None
    
    def test_sobol_result(self):
        """Test Sobol sensitivity result."""
        result = SensitivityResult(
            parameter="param1",
            method="sobol",
            mu_star=0.0,
            sigma=0.0,
            sobol_first=0.3,
            sobol_total=0.5,
            confidence_interval=(0.0, 0.0),
        )
        
        assert result.parameter == "param1"
        assert result.method == "sobol"
        assert result.sobol_first == 0.3
        assert result.sobol_total == 0.5
        assert result.sobol_first <= result.sobol_total


class TestCreateEnsembleEngine:
    """Tests for create_ensemble_engine factory function."""
    
    def test_factory(self):
        """Test factory function creates engine with config."""
        config = {
            "simulation": {"seed": 123},
            "ensemble": {"n_workers": 4},
        }
        
        from engine.ensemble import create_ensemble_engine
        engine = create_ensemble_engine(config)
        
        assert engine.base_seed == 123
        assert engine.n_workers == 4
    
    def test_factory_defaults(self):
        """Test factory with missing config sections."""
        config = {}
        
        from engine.ensemble import create_ensemble_engine
        engine = create_ensemble_engine(config)
        
        assert engine.base_seed == 42  # Default
        assert engine.n_workers == 4  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])