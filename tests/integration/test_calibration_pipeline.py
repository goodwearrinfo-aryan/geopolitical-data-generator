"""Integration tests for calibration pipeline."""

from __future__ import annotations

import pytest
import subprocess
import tempfile
import json
from pathlib import Path


class TestCalibrationPipeline:
    """Integration tests for calibration pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    
    def test_calibrate_moment_matching(self, temp_dir):
        """Test calibration with moment matching method."""
        result = subprocess.run([
            "python3", "cli.py", "calibrate",
            "--data-dir", "tests/fixtures",
            "--method", "moment_matching",
            "--output", str(temp_dir / "calibration"),
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        # Check output file
        assert (temp_dir / "calibration" / "calibration_result.json").exists()
        
        with open(temp_dir / "calibration" / "calibration_result.json") as f:
            result_data = json.load(f)
        
        assert "parameters" in result_data
        assert "objective" in result_data
        assert "convergence" in result_data
        assert "iterations" in result_data
    
    def test_calibrate_hybrid(self, temp_dir):
        """Test calibration with hybrid method."""
        result = subprocess.run([
            "python3", "cli.py", "calibrate",
            "--data-dir", "tests/fixtures",
            "--method", "hybrid",
            "--output", str(temp_dir / "calibration"),
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        assert (temp_dir / "calibration" / "calibration_result.json").exists()
    
    def test_calibrate_with_validation(self, temp_dir):
        """Test calibration with validation flag."""
        result = subprocess.run([
            "python3", "cli.py", "calibrate",
            "--data-dir", "tests/fixtures",
            "--method", "moment_matching",
            "--output", str(temp_dir / "calibration"),
            "--validate",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0, f"stderr: {result.stderr}"
        
        # Validation metrics should be printed
        assert "Validation metrics:" in result.stdout


class TestCalibrationFixtures:
    """Tests for calibration fixtures."""
    
    def test_wdi_fixtures_exist(self):
        """Test that WDI fixtures were generated."""
        wdi_dir = Path("tests/fixtures/world_bank/v1")
        assert wdi_dir.exists()
        
        required_files = [
            "gdp.csv",
            "population.csv",
            "trade.csv",
            "economic_indicators.csv",
            "energy.csv",
            "countries.csv",
        ]
        for f in required_files:
            assert (wdi_dir / f).exists(), f"Missing {f}"
    
    def test_inscr_fixtures_exist(self):
        """Test that INSCR fixtures were generated."""
        inscr_dir = Path("tests/fixtures/inscr/v1")
        assert inscr_dir.exists()
        
        required_files = [
            "polity.csv",
            "coups.csv",
            "state_fragility.csv",
            "mepv.csv",
            "displaced_populations.csv",
            "igo_memberships.csv",
        ]
        for f in required_files:
            assert (inscr_dir / f).exists(), f"Missing {f}"
    
    def test_acled_fixtures_exist(self):
        """Test that ACLED fixtures were generated."""
        acled_dir = Path("tests/fixtures/acled/v1")
        assert acled_dir.exists()
        
        required_files = [
            "events.csv",
            "actors.csv",
            "event_types.csv",
        ]
        for f in required_files:
            assert (acled_dir / f).exists(), f"Missing {f}"
    
    def test_fixture_data_quality(self):
        """Test that fixture data has expected quality."""
        import pandas as pd
        
        # Check WDI GDP data
        gdp = pd.read_csv("tests/fixtures/world_bank/v1/gdp.csv")
        assert len(gdp) > 0
        assert "iso3" in gdp.columns
        assert "year" in gdp.columns
        assert "gdp_usd" in gdp.columns
        assert gdp["gdp_usd"].min() > 0
        
        # Check INSCR polity data
        polity = pd.read_csv("tests/fixtures/inscr/v1/polity.csv")
        assert len(polity) > 0
        assert "iso3" in polity.columns
        assert "polity2" in polity.columns
        assert polity["polity2"].between(-10, 10).all()
        
        # Check ACLED events
        events = pd.read_csv("tests/fixtures/acled/v1/events.csv")
        assert len(events) > 0
        assert "event_type" in events.columns
        assert "fatalities" in events.columns
        assert events["fatalities"].min() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])