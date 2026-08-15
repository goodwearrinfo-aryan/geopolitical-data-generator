"""Integration tests for full simulation pipeline."""

from __future__ import annotations

import pytest
import subprocess
import tempfile
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path


class TestFullPipeline:
    """Integration tests for the full simulation pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    
    def test_run_single_simulation_parquet(self, temp_dir):
        """Test running a single simulation with Parquet output."""
        result = subprocess.run([
            "python3", "cli.py", "run",
            "--timesteps", "6",
            "--output", str(temp_dir),
            "--format", "parquet",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        # Check output files
        countries_file = temp_dir / "countries.parquet"
        summary_file = temp_dir / "summary.parquet"
        
        assert countries_file.exists()
        assert summary_file.exists()
        
        # Validate countries parquet
        table = pq.read_table(countries_file)
        assert len(table) == 50  # 50 core countries
        assert "regime_type" in table.column_names
        assert "iso3" in table.column_names
        assert "population" in table.column_names
        assert "gdp_usd" in table.column_names
        
        # Check regime types are valid
        regimes = set(table["regime_type"].to_pylist())
        assert regimes.issubset({"democracy", "autocracy", "anocracy", "failed_state"})
        
        # Validate summary
        summary_table = pq.read_table(summary_file)
        assert len(summary_table) == 1
        assert summary_table["n_countries"][0].as_py() == 50
    
    def test_run_single_simulation_csv(self, temp_dir):
        """Test running a single simulation with CSV output."""
        result = subprocess.run([
            "python3", "cli.py", "run",
            "--timesteps", "3",
            "--output", str(temp_dir),
            "--format", "csv",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        countries_file = temp_dir / "countries.csv"
        assert countries_file.exists()
        
        df = pd.read_csv(countries_file)
        assert len(df) == 50
        assert "regime_type" in df.columns
        assert df["iso3"].nunique() == 50
    
    def test_run_multiple_formats(self, temp_dir):
        """Test running with multiple export formats."""
        result = subprocess.run([
            "python3", "cli.py", "run",
            "--timesteps", "2",
            "--output", str(temp_dir),
            "--format", "parquet",
            "--format", "csv",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        assert (temp_dir / "countries.parquet").exists()
        assert (temp_dir / "countries.csv").exists()
    
    def test_run_with_custom_seed(self, temp_dir):
        """Test running with custom seed for reproducibility."""
        # Run twice with same seed
        result1 = subprocess.run([
            "python3", "cli.py", "run",
            "--timesteps", "3",
            "--output", str(temp_dir / "run1"),
            "--format", "parquet",
            "--seed", "12345",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        result2 = subprocess.run([
            "python3", "cli.py", "run",
            "--timesteps", "3",
            "--output", str(temp_dir / "run2"),
            "--format", "parquet",
            "--seed", "12345",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result1.returncode == 0
        assert result2.returncode == 0
        
        # Results should be identical
        table1 = pq.read_table(temp_dir / "run1" / "countries.parquet")
        table2 = pq.read_table(temp_dir / "run2" / "countries.parquet")
        
        # Compare GDP values (should be identical with same seed)
        gdp1 = table1["gdp_usd"].to_pylist()
        gdp2 = table2["gdp_usd"].to_pylist()
        assert gdp1 == gdp2
    
    def test_simulation_progress(self, temp_dir):
        """Test that simulation progresses through timesteps."""
        result = subprocess.run([
            "python3", "cli.py", "run",
            "--timesteps", "12",
            "--output", str(temp_dir),
            "--format", "parquet",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0
        
        # Check final timestep in summary
        summary_table = pq.read_table(temp_dir / "summary.parquet")
        assert summary_table["timestep"][0].as_py() == 12


class TestEnsemblePipeline:
    """Integration tests for ensemble simulation pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    
    def test_ensemble_small(self, temp_dir):
        """Test running a small ensemble."""
        result = subprocess.run([
            "python3", "cli.py", "ensemble",
            "--scenarios", "3",
            "--output", str(temp_dir / "ensemble"),
            "--workers", "2",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        # Check output files
        assert (temp_dir / "ensemble" / "ensemble_metrics.csv").exists()
        assert (temp_dir / "ensemble" / "ensemble_metrics.parquet").exists()
        assert (temp_dir / "ensemble" / "ensemble_summary.json").exists()
        
        # Validate metrics
        df = pd.read_csv(temp_dir / "ensemble" / "ensemble_metrics.csv")
        assert len(df) == 3
        assert "run_id" in df.columns
        assert "final_global_gdp" in df.columns
        assert "peak_conflicts" in df.columns
    
    def test_ensemble_with_sensitivity(self, temp_dir):
        """Test ensemble with sensitivity analysis."""
        result = subprocess.run([
            "python3", "cli.py", "ensemble",
            "--scenarios", "5",
            "--output", str(temp_dir / "ensemble"),
            "--workers", "2",
            "--sensitivity",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        # Check sensitivity results
        assert (temp_dir / "ensemble" / "sensitivity_results.csv").exists()
        assert (temp_dir / "ensemble" / "sensitivity_results.parquet").exists()
        
        df = pd.read_csv(temp_dir / "ensemble" / "sensitivity_results.csv")
        assert len(df) > 0
        assert "parameter" in df.columns
        assert "method" in df.columns
        assert "mu_star" in df.columns
        assert "sigma" in df.columns


class TestCalibrationPipeline:
    """Integration tests for calibration pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    
    def test_calibrate_moment_matching(self, temp_dir):
        """Test calibration with moment matching."""
        # This test uses the synthetic fixtures
        result = subprocess.run([
            "python3", "cli.py", "calibrate",
            "--data-dir", "tests/fixtures",
            "--method", "moment_matching",
            "--output", str(temp_dir / "calibration"),
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        # Should run without error (may not converge perfectly with synthetic data)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        # Check output
        assert (temp_dir / "calibration" / "calibration_result.json").exists()
    
    def test_validate_command(self):
        """Test validate command."""
        result = subprocess.run([
            "python3", "cli.py", "validate",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0
        assert "Validation passed!" in result.stdout
        assert "Simulation: 2020-2050" in result.stdout


class TestInitConfig:
    """Integration tests for init-config command."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    
    def test_init_config(self, temp_dir):
        """Test generating default config file."""
        config_file = temp_dir / "config.yaml"
        
        result = subprocess.run([
            "python3", "cli.py", "init-config",
            "--output", str(config_file),
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0
        assert config_file.exists()
        
        # Validate config content
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        assert "simulation" in config
        assert "calibration" in config
        assert "domains" in config
        assert "ensemble" in config
        assert "weak_signals" in config


class TestCLIErrorHandling:
    """Tests for CLI error handling."""
    
    def test_invalid_format(self):
        """Test that invalid format gives error."""
        result = subprocess.run([
            "python3", "cli.py", "run",
            "--format", "invalid",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode != 0
        assert "invalid" in result.stderr.lower()
    
    def test_missing_required_args(self):
        """Test missing required arguments."""
        result = subprocess.run([
            "python3", "cli.py", "ensemble",
            # Missing --output
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        # ensemble has default output, so this might succeed
        # Test a command that requires args
        result = subprocess.run([
            "python3", "cli.py", "calibrate",
            # Missing --data-dir
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])