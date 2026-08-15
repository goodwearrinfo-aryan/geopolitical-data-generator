"""Integration tests for ensemble pipeline."""

from __future__ import annotations

import pytest
import subprocess
import tempfile
import json
import pandas as pd
from pathlib import Path


class TestEnsemblePipeline:
    """Integration tests for ensemble simulation pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    
    def test_ensemble_basic(self, temp_dir):
        """Test basic ensemble run."""
        result = subprocess.run([
            "python3", "cli.py", "ensemble",
            "--scenarios", "3",
            "--output", str(temp_dir / "ensemble"),
            "--workers", "2",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        # Check all expected output files
        ensemble_dir = temp_dir / "ensemble"
        assert (ensemble_dir / "ensemble_metrics.csv").exists()
        assert (ensemble_dir / "ensemble_metrics.parquet").exists()
        assert (ensemble_dir / "ensemble_summary.json").exists()
        
        # Validate metrics CSV
        df = pd.read_csv(ensemble_dir / "ensemble_metrics.csv")
        assert len(df) == 3
        assert list(df.columns) == [
            "run_id", "seed", "final_global_gdp", "avg_growth_rate",
            "peak_conflicts", "final_conflicts", "gdp_volatility", "conflict_volatility"
        ]
        
        # Validate summary JSON
        with open(ensemble_dir / "ensemble_summary.json") as f:
            summary = json.load(f)
        
        assert "final_global_gdp" in summary
        assert "mean" in summary["final_global_gdp"]
        assert "std" in summary["final_global_gdp"]
        assert "gdp_trajectory" in summary
        assert "conflict_trajectory" in summary
    
    def test_ensemble_sensitivity_morris(self, temp_dir):
        """Test ensemble with Morris sensitivity analysis."""
        result = subprocess.run([
            "python3", "cli.py", "ensemble",
            "--scenarios", "5",
            "--output", str(temp_dir / "ensemble"),
            "--workers", "2",
            "--sensitivity",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        ensemble_dir = temp_dir / "ensemble"
        assert (ensemble_dir / "sensitivity_results.csv").exists()
        assert (ensemble_dir / "sensitivity_results.parquet").exists()
        
        df = pd.read_csv(ensemble_dir / "sensitivity_results.csv")
        assert len(df) == 10  # 5 default parameters * 2 methods (morris + sobol)
        morris_rows = df[df["method"] == "morris"]
        assert len(morris_rows) == 5
        assert all(morris_rows["mu_star"] >= 0)
        assert all(morris_rows["sigma"] >= 0)
    
    def test_ensemble_reproducibility(self, temp_dir):
        """Test ensemble reproducibility with fixed seed."""
        # The ensemble uses base_seed from config (default 42)
        # Running twice should produce same results
        
        result1 = subprocess.run([
            "python3", "cli.py", "ensemble",
            "--scenarios", "3",
            "--output", str(temp_dir / "ens1"),
            "--workers", "1",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        result2 = subprocess.run([
            "python3", "cli.py", "ensemble",
            "--scenarios", "3",
            "--output", str(temp_dir / "ens2"),
            "--workers", "1",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result1.returncode == 0
        assert result2.returncode == 0
        
        # Compare summary statistics
        with open(temp_dir / "ens1" / "ensemble_summary.json") as f:
            summary1 = json.load(f)
        with open(temp_dir / "ens2" / "ensemble_summary.json") as f:
            summary2 = json.load(f)
        
        # Mean GDP should be identical
        assert summary1["final_global_gdp"]["mean"] == summary2["final_global_gdp"]["mean"]
    
    def test_ensemble_larger_scale(self, temp_dir):
        """Test ensemble with more scenarios."""
        result = subprocess.run([
            "python3", "cli.py", "ensemble",
            "--scenarios", "10",
            "--output", str(temp_dir / "ensemble"),
            "--workers", "4",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        ensemble_dir = temp_dir / "ensemble"
        df = pd.read_csv(ensemble_dir / "ensemble_metrics.csv")
        assert len(df) == 10
        assert df["run_id"].nunique() == 10
        assert df["seed"].nunique() == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])