"""Unit tests for exporters module."""

from __future__ import annotations

import pytest
import pandas as pd
import pyarrow.parquet as pq
from datetime import date
from pathlib import Path
from uuid import UUID

from exporters import (
    ParquetExporter, CSVExporter, GeoJSONExporter, NetworkExporter,
    get_exporter, BaseExporter, _enum_value
)
from engine.temporal import TemporalEngine, TimestepFrequency
from schemas.core import (
    SimulationState, Country, PoliticalEvent, EventType, Conflict,
    ConflictType, ConflictIntensity, Alliance, AllianceType, Treaty,
    TreatyCategory, Sanction, SanctionType, Leader, RegimeType,
    CountryTier, ResourceType, EconomicIndicator, TradeFlow,
    MigrationFlow, DemographicProfile
)


class TestEnumValue:
    """Tests for _enum_value helper."""
    
    def test_enum_input(self):
        """Test with enum input."""
        result = _enum_value(RegimeType.DEMOCRACY)
        assert result == "democracy"
    
    def test_string_input(self):
        """Test with string input (already a value)."""
        result = _enum_value("democracy")
        assert result == "democracy"
    
    def test_none_input(self):
        """Test with None input."""
        result = _enum_value(None)
        assert result is None


class TestParquetExporter:
    """Tests for ParquetExporter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.output_dir = "./test_output_parquet"
        self.exporter = ParquetExporter(self.output_dir)
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def test_export_creates_files(self):
        """Test that export creates expected files."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        assert Path(self.output_dir, "countries.parquet").exists()
        assert Path(self.output_dir, "summary.parquet").exists()
        # Other files only created if data exists
    
    def test_countries_parquet_schema(self):
        """Test countries parquet has correct schema."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        table = pq.read_table(Path(self.output_dir, "countries.parquet"))
        
        assert len(table) == 2  # Two test countries
        expected_cols = {
            "timestep", "date", "iso3", "iso2", "name", "region",
            "regime_type", "population", "gdp_usd", "stability_index"
        }
        assert expected_cols.issubset(set(table.column_names))
        
        # Check regime_type is string (not enum)
        regimes = table["regime_type"].to_pylist()
        assert all(isinstance(r, str) for r in regimes)
        assert set(regimes) == {"democracy", "autocracy"}
    
    def test_summary_parquet(self):
        """Test summary parquet file."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        table = pq.read_table(Path(self.output_dir, "summary.parquet"))
        
        assert len(table) == 1
        assert table["timestep"][0].as_py() == 5
        assert table["n_countries"][0].as_py() == 2
    
    def _create_test_state(self) -> SimulationState:
        """Create a minimal test state."""
        state = SimulationState(timestep=5, date=date(2020, 6, 1))
        
        usa = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.8,
            gdp_growth_rate=0.025,
        )
        chn = Country(
            iso3="CHN", iso2="CN", name="China",
            region="Asia", subregion="Eastern Asia",
            area_km2=9596961, population=1412000000,
            gdp_usd=18e12, gdp_per_capita_usd=12000,
            regime_type=RegimeType.AUTOCRACY,
            stability_index=0.6,
            gdp_growth_rate=0.05,
        )
        
        state.countries["USA"] = usa
        state.countries["CHN"] = chn
        state.global_gdp_usd = 43e12
        state.global_population = 1743000000
        
        return state


class TestCSVExporter:
    """Tests for CSVExporter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.output_dir = "./test_output_csv"
        self.exporter = CSVExporter(self.output_dir)
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def test_export_creates_csv_files(self):
        """Test that export creates CSV files."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        assert Path(self.output_dir, "countries.csv").exists()
        # CSV exporter doesn't export summary.csv
    
    def test_countries_csv_content(self):
        """Test countries CSV content."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        df = pd.read_csv(Path(self.output_dir, "countries.csv"))
        
        assert len(df) == 2
        assert list(df["iso3"]) == ["USA", "CHN"]
        assert list(df["regime_type"]) == ["democracy", "autocracy"]
        assert df["population"].tolist() == [331000000, 1412000000]
    
    def _create_test_state(self) -> SimulationState:
        """Create a minimal test state."""
        state = SimulationState(timestep=5, date=date(2020, 6, 1))
        
        usa = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.8,
            gdp_growth_rate=0.025,
        )
        chn = Country(
            iso3="CHN", iso2="CN", name="China",
            region="Asia", subregion="Eastern Asia",
            area_km2=9596961, population=1412000000,
            gdp_usd=18e12, gdp_per_capita_usd=12000,
            regime_type=RegimeType.AUTOCRACY,
            stability_index=0.6,
            gdp_growth_rate=0.05,
        )
        
        state.countries["USA"] = usa
        state.countries["CHN"] = chn
        
        return state


class TestGeoJSONExporter:
    """Tests for GeoJSONExporter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.output_dir = "./test_output_geojson"
        self.exporter = GeoJSONExporter(self.output_dir)
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def test_export_creates_geojson(self):
        """Test that export creates GeoJSON file."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        geojson_files = list(Path(self.output_dir).glob("*.geojson"))
        assert len(geojson_files) >= 1
    
    def test_geojson_structure(self):
        """Test GeoJSON structure."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        import json
        geojson_files = list(Path(self.output_dir).glob("*.geojson"))
        with open(geojson_files[0]) as f:
            data = json.load(f)
        
        assert data["type"] == "FeatureCollection"
        assert "features" in data
        assert len(data["features"]) == 2
        
        feature = data["features"][0]
        assert feature["type"] == "Feature"
        assert "properties" in feature
        assert "geometry" in feature
        assert feature["properties"]["iso3"] in ["USA", "CHN"]
        assert feature["geometry"]["type"] == "Point"
    
    def _create_test_state(self) -> SimulationState:
        """Create a minimal test state."""
        state = SimulationState(timestep=5, date=date(2020, 6, 1))
        
        usa = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.8,
        )
        chn = Country(
            iso3="CHN", iso2="CN", name="China",
            region="Asia", subregion="Eastern Asia",
            area_km2=9596961, population=1412000000,
            gdp_usd=18e12, gdp_per_capita_usd=12000,
            regime_type=RegimeType.AUTOCRACY,
            stability_index=0.6,
        )
        
        state.countries["USA"] = usa
        state.countries["CHN"] = chn
        
        return state


class TestNetworkExporter:
    """Tests for NetworkExporter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.output_dir = "./test_output_network"
        self.exporter = NetworkExporter(self.output_dir)
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def test_export_creates_network_files(self):
        """Test that export creates network files."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        assert Path(self.output_dir, "diplomatic_network_t5.graphml").exists()
        assert Path(self.output_dir, "trade_network_t5.graphml").exists()
        assert Path(self.output_dir, "alliance_network_t5.graphml").exists()
        assert Path(self.output_dir, "diplomatic_network_t5.json").exists()
        assert Path(self.output_dir, "trade_network_t5.json").exists()
        assert Path(self.output_dir, "alliance_network_t5.json").exists()
    
    def test_graphml_readable(self):
        """Test that GraphML files are readable."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        import networkx as nx
        G = nx.read_graphml(Path(self.output_dir, "diplomatic_network_t5.graphml"))
        
        assert G.number_of_nodes() == 2
        assert G.has_node("USA")
        assert G.has_node("CHN")
        # GraphML may not preserve all attributes
        # Just verify the graph structure
    
    def test_json_readable(self):
        """Test that JSON network files are readable."""
        state = self._create_test_state()
        
        self.exporter.export(state)
        
        import json
        with open(Path(self.output_dir, "diplomatic_network_t5.json")) as f:
            data = json.load(f)
        
        assert "nodes" in data
        assert "edges" in data  # node_link_data uses 'edges' not 'links'
        assert len(data["nodes"]) == 2
    
    def _create_test_state(self) -> SimulationState:
        """Create a minimal test state."""
        state = SimulationState(timestep=5, date=date(2020, 6, 1))
        
        usa = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.8,
            trade_partners={"CHN": 5e11},
            diplomatic_relations={"CHN": 50},
        )
        chn = Country(
            iso3="CHN", iso2="CN", name="China",
            region="Asia", subregion="Eastern Asia",
            area_km2=9596961, population=1412000000,
            gdp_usd=18e12, gdp_per_capita_usd=12000,
            regime_type=RegimeType.AUTOCRACY,
            stability_index=0.6,
            trade_partners={"USA": 4e11},
            diplomatic_relations={"USA": 50},
        )
        
        state.countries["USA"] = usa
        state.countries["CHN"] = chn
        
        # Add alliance
        alliance = Alliance(
            name="Test Alliance",
            alliance_type=AllianceType.DEFENSE,
            members=["USA", "CHN"],
            founding_date=date(2020, 1, 1),
            is_active=True,
        )
        state.alliances[alliance.id] = alliance
        
        return state


class TestGetExporter:
    """Tests for get_exporter factory function."""
    
    def test_get_parquet_exporter(self):
        """Test getting ParquetExporter."""
        exporter = get_exporter("parquet", "./test")
        assert isinstance(exporter, ParquetExporter)
    
    def test_get_csv_exporter(self):
        """Test getting CSVExporter."""
        exporter = get_exporter("csv", "./test")
        assert isinstance(exporter, CSVExporter)
    
    def test_get_geojson_exporter(self):
        """Test getting GeoJSONExporter."""
        exporter = get_exporter("geojson", "./test")
        assert isinstance(exporter, GeoJSONExporter)
    
    def test_get_network_exporter(self):
        """Test getting NetworkExporter."""
        exporter = get_exporter("network", "./test")
        assert isinstance(exporter, NetworkExporter)
    
    def test_get_kafka_exporter(self):
        """Test getting KafkaExporter."""
        exporter = get_exporter("kafka", "./test", bootstrap_servers="localhost:9092")
        assert isinstance(exporter, type(exporter))
        assert exporter.__class__.__name__ == "KafkaExporter"
    
    def test_get_neo4j_exporter(self):
        """Test getting Neo4jExporter."""
        exporter = get_exporter("neo4j", "./test", uri="bolt://localhost:7687")
        assert isinstance(exporter, type(exporter))
        assert exporter.__class__.__name__ == "Neo4jExporter"
    
    def test_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            get_exporter("invalid_format", "./test")
    
    def test_kafka_exporter_kwargs(self):
        """Test KafkaExporter accepts kwargs."""
        exporter = get_exporter(
            "kafka", "./test",
            bootstrap_servers="kafka:9092",
            topic_prefix="test",
            flush_interval=50,
        )
        assert exporter.bootstrap_servers == "kafka:9092"
        assert exporter.topic_prefix == "test"
        assert exporter.flush_interval == 50
    
    def test_neo4j_exporter_kwargs(self):
        """Test Neo4jExporter accepts kwargs."""
        exporter = get_exporter(
            "neo4j", "./test",
            uri="bolt://neo4j:7687",
            user="neo4j",
            password="secret",
            batch_size=500,
        )
        assert exporter.uri == "bolt://neo4j:7687"
        assert exporter.user == "neo4j"
        assert exporter.password == "secret"
        assert exporter.batch_size == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])