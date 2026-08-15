"""Integration tests for streaming pipeline (Kafka, Neo4j)."""

from __future__ import annotations

import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestStreamingPipeline:
    """Integration tests for streaming pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    
    @pytest.mark.streaming
    def test_kafka_exporter_import(self):
        """Test that KafkaExporter can be imported."""
        from exporters import KafkaExporter
        assert KafkaExporter is not None
    
    @pytest.mark.streaming
    def test_neo4j_exporter_import(self):
        """Test that Neo4jExporter can be imported."""
        from exporters import Neo4jExporter
        assert Neo4jExporter is not None
    
    @pytest.mark.streaming
    def test_kafka_exporter_initialization(self):
        """Test KafkaExporter initialization with custom params."""
        from exporters import KafkaExporter
        
        exporter = KafkaExporter(
            "./test_output",
            bootstrap_servers="kafka:9092",
            topic_prefix="test",
            flush_interval=50,
        )
        
        assert exporter.bootstrap_servers == "kafka:9092"
        assert exporter.topic_prefix == "test"
        assert exporter.flush_interval == 50
    
    @pytest.mark.streaming
    def test_neo4j_exporter_initialization(self):
        """Test Neo4jExporter initialization with custom params."""
        from exporters import Neo4jExporter
        
        exporter = Neo4jExporter(
            "./test_output",
            uri="bolt://neo4j:7687",
            user="neo4j",
            password="secret",
            batch_size=500,
        )
        
        assert exporter.uri == "bolt://neo4j:7687"
        assert exporter.user == "neo4j"
        assert exporter.password == "secret"
        assert exporter.batch_size == 500
    
    @pytest.mark.streaming
    def test_cli_stream_flag_exists(self):
        """Test that --stream flag exists in CLI."""
        result = subprocess.run([
            "python3", "cli.py", "run", "--help",
        ], capture_output=True, text=True, cwd="/tmp/geopolitical-data-generator")
        
        assert result.returncode == 0
        assert "--stream" in result.stdout
        assert "kafka" in result.stdout
        assert "neo4j" in result.stdout
    
    @pytest.mark.streaming
    def test_get_exporter_kafka(self):
        """Test factory function for Kafka exporter."""
        from exporters import get_exporter
        
        exporter = get_exporter("kafka", "./test", bootstrap_servers="localhost:9092")
        assert exporter.__class__.__name__ == "KafkaExporter"
        assert exporter.bootstrap_servers == "localhost:9092"
    
    @pytest.mark.streaming
    def test_get_exporter_neo4j(self):
        """Test factory function for Neo4j exporter."""
        from exporters import get_exporter
        
        exporter = get_exporter("neo4j", "./test", uri="bolt://localhost:7687")
        assert exporter.__class__.__name__ == "Neo4jExporter"
        assert exporter.uri == "bolt://localhost:7687"
    
    # Mock-based tests for when services aren't available
    @pytest.mark.streaming
    def test_kafka_export_with_mock(self):
        """Test Kafka export with mocked producer."""
        from exporters import KafkaExporter
        from schemas.core import SimulationState, Country, RegimeType
        from datetime import date
        
        with patch('exporters.KafkaExporter._get_producer') as mock_get_producer:
            mock_producer = MagicMock()
            mock_get_producer.return_value = mock_producer
            
            exporter = KafkaExporter("./test", bootstrap_servers="mock:9092")
            
            state = SimulationState(timestep=1, date=date(2020, 1, 1))
            country = Country(
                iso3="USA", iso2="US", name="United States",
                region="Americas", subregion="Northern America",
                area_km2=9833517, population=331000000,
                gdp_usd=25e12, gdp_per_capita_usd=75000,
                regime_type=RegimeType.DEMOCRACY,
            )
            state.countries["USA"] = country
            
            exporter.export(state)
            
            # Verify produce was called
            assert mock_producer.produce.called
    
    @pytest.mark.streaming
    def test_neo4j_export_with_mock(self):
        """Test Neo4j export with mocked driver."""
        from exporters import Neo4jExporter
        from schemas.core import SimulationState, Country, RegimeType
        from datetime import date
        
        with patch('exporters.Neo4jExporter._get_driver') as mock_get_driver:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_get_driver.return_value = mock_driver
            
            exporter = Neo4jExporter("./test", uri="bolt://mock:7687")
            
            state = SimulationState(timestep=1, date=date(2020, 1, 1))
            country = Country(
                iso3="USA", iso2="US", name="United States",
                region="Americas", subregion="Northern America",
                area_km2=9833517, population=331000000,
                gdp_usd=25e12, gdp_per_capita_usd=75000,
                regime_type=RegimeType.DEMOCRACY,
            )
            state.countries["USA"] = country
            
            exporter.export(state)
            
            # Verify session.execute_write was called
            assert mock_session.execute_write.called


class TestStreamingConfig:
    """Tests for streaming configuration."""
    
    def test_config_has_streaming_section(self):
        """Test that default config has streaming section."""
        import yaml
        with open("configs/default.yaml") as f:
            config = yaml.safe_load(f)
        
        # The config should have streaming-related settings
        assert "simulation" in config
        assert "kafka_enabled" in config["simulation"]
        assert "kafka_bootstrap" in config["simulation"]
        assert "kafka_topic_prefix" in config["simulation"]
        assert "neo4j_enabled" in config["simulation"]
        assert "neo4j_uri" in config["simulation"]
    
    def test_config_loader_loads_streaming(self):
        """Test that config loader loads streaming settings."""
        from config.loader import load_config, SimulationConfig
        
        config = load_config("configs/default.yaml")
        
        assert hasattr(config.simulation, "kafka_enabled")
        assert hasattr(config.simulation, "kafka_bootstrap")
        assert hasattr(config.simulation, "kafka_topic_prefix")
        assert hasattr(config.simulation, "neo4j_enabled")
        assert hasattr(config.simulation, "neo4j_uri")
        assert hasattr(config.simulation, "neo4j_user")
        assert hasattr(config.simulation, "neo4j_password")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])