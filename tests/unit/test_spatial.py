"""Unit tests for engine.spatial module."""

from __future__ import annotations

from datetime import date
import pytest
import numpy as np
import networkx as nx

from engine.spatial import SpatialEngine, CountryGeometry, build_spatial_graph
from engine.temporal import TemporalEngine, TimestepFrequency
from schemas.core import SimulationState, Country, ResourceType, RegimeType, CountryTier


class TestSpatialEngine:
    """Tests for SpatialEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SpatialEngine({})
    
    def test_initialization(self):
        """Test spatial engine initialization."""
        assert self.engine.geometries is not None
        assert len(self.engine.geometries) > 0
        assert self.engine.distance_matrix is not None
        assert isinstance(self.engine.trade_graph, nx.DiGraph)
        assert isinstance(self.engine.contagion_graph, nx.Graph)
    
    def test_get_distance_same_country(self):
        """Test distance between same country is 0."""
        dist = self.engine.get_distance("USA", "USA")
        assert dist == 0.0
    
    def test_get_distance_known_countries(self):
        """Test distance between known countries."""
        # USA to CAN should be relatively small
        dist = self.engine.get_distance("USA", "CAN")
        assert 0 < dist < 5000
        
        # USA to CHN should be large
        dist = self.engine.get_distance("USA", "CHN")
        assert dist > 10000
    
    def test_get_distance_unknown_countries(self):
        """Test distance for unknown countries returns default."""
        dist = self.engine.get_distance("XXX", "YYY")
        assert dist == 20000.0  # Default ~half earth circumference
    
    def test_are_neighbors(self):
        """Test neighbor detection."""
        # USA and CAN are neighbors
        assert self.engine.are_neighbors("USA", "CAN") is True
        assert self.engine.are_neighbors("CAN", "USA") is True
        
        # USA and CHN are not neighbors
        assert self.engine.are_neighbors("USA", "CHN") is False
    
    def test_get_neighbors(self):
        """Test getting neighbor list."""
        neighbors = self.engine.get_neighbors("USA")
        
        assert "CAN" in neighbors
        assert "MEX" in neighbors
        assert len(neighbors) >= 2
    
    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # Known distance: NYC to LA ~3940 km
        # NYC: 40.7, -74.0; LA: 34.1, -118.2
        dist = self.engine._haversine(40.7, -74.0, 34.1, -118.2)
        assert 3500 < dist < 4500
    
    def test_calculate_trade_gravity(self):
        """Test trade gravity model calculation."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        exporter = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            trade_openness=1.0,
            resources={ResourceType.OIL: 1000000},
            sanctions_on=[],
        )
        
        importer = Country(
            iso3="CAN", iso2="CA", name="Canada",
            region="Americas", subregion="Northern America",
            area_km2=9984670, population=38000000,
            gdp_usd=2e12, gdp_per_capita_usd=52000,
            regime_type=RegimeType.DEMOCRACY,
            trade_openness=1.0,
            resources={},
            sanctions_on=[],
        )
        
        state.countries["USA"] = exporter
        state.countries["CAN"] = importer
        
        flow = self.engine.calculate_trade_gravity(exporter, importer, state)
        
        assert flow > 0
        # USA-CAN should have high trade due to proximity and similar GDP
        assert flow > 1e10
    
    def test_calculate_trade_gravity_with_sanctions(self):
        """Test trade gravity with sanctions."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        exporter = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            sanctions_on=["IRN"],
        )
        
        importer = Country(
            iso3="IRN", iso2="IR", name="Iran",
            region="Asia", subregion="Southern Asia",
            area_km2=1648195, population=84000000,
            gdp_usd=5e11, gdp_per_capita_usd=6000,
            regime_type=RegimeType.AUTOCRACY,
            sanctions_by=["USA"],
        )
        
        state.countries["USA"] = exporter
        state.countries["IRN"] = importer
        
        # Calculate flow without sanctions (temporarily remove sanctions)
        exporter_no_sanctions = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            sanctions_on=[],
        )
        importer_no_sanctions = Country(
            iso3="IRN", iso2="IR", name="Iran",
            region="Asia", subregion="Southern Asia",
            area_km2=1648195, population=84000000,
            gdp_usd=5e11, gdp_per_capita_usd=6000,
            regime_type=RegimeType.AUTOCRACY,
            sanctions_by=[],
        )
        
        flow_with_sanctions = self.engine.calculate_trade_gravity(exporter, importer, state)
        
        # Create temp state for no-sanctions comparison
        temp_state = SimulationState(timestep=0, date=date(2020, 1, 1))
        temp_state.countries["USA"] = exporter_no_sanctions
        temp_state.countries["IRN"] = importer_no_sanctions
        flow_without_sanctions = self.engine.calculate_trade_gravity(exporter_no_sanctions, importer_no_sanctions, temp_state)
        
        # Sanctions should reduce trade significantly (at least 50% reduction)
        assert flow_with_sanctions < flow_without_sanctions * 0.5
        assert flow_with_sanctions < flow_without_sanctions
    
    def test_calculate_conflict_contagion(self):
        """Test conflict contagion probability."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        # Add conflict
        from schemas.core import Conflict, ConflictType, ConflictIntensity
        conflict = Conflict(
            name="Test War",
            conflict_type=ConflictType.INTERSTATE,
            intensity=ConflictIntensity.MAJOR,
            primary_attacker="USA",
            primary_defender="CAN",
            start_date=date(2020, 1, 1),
            displaced_persons=100000,
            theater_countries=["USA", "CAN"],
        )
        state.conflicts[conflict.id] = conflict
        
        # Add countries
        usa = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        can = Country(
            iso3="CAN", iso2="CA", name="Canada",
            region="Americas", subregion="Northern America",
            area_km2=9984670, population=38000000,
            gdp_usd=2e12, gdp_per_capita_usd=52000,
            regime_type=RegimeType.DEMOCRACY,
        )
        state.countries["USA"] = usa
        state.countries["CAN"] = can
        
        # USA-CAN are neighbors, should have contagion risk
        prob = self.engine.calculate_conflict_contagion("USA", "CAN", state)
        assert prob > 0
        
        # Non-neighbors should have zero risk
        prob = self.engine.calculate_conflict_contagion("USA", "CHN", state)
        assert prob == 0.0
    
    def test_calculate_migration_potential(self):
        """Test migration potential calculation."""
        origin = Country(
            iso3="SYR", iso2="SY", name="Syria",
            region="Asia", subregion="Western Asia",
            area_km2=185180, population=17000000,
            gdp_usd=2e10, gdp_per_capita_usd=1200,
            regime_type=RegimeType.AUTOCRACY,
            stability_index=0.2,
            gdp_growth_rate=-0.1,
            unemployment_rate=0.5,
        )
        
        destination = Country(
            iso3="DEU", iso2="DE", name="Germany",
            region="Europe", subregion="Western Europe",
            area_km2=357114, population=83000000,
            gdp_usd=4e12, gdp_per_capita_usd=48000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.8,
            gdp_growth_rate=0.02,
            unemployment_rate=0.04,
        )
        
        potential = self.engine.calculate_migration_potential(origin, destination)
        
        assert potential > 0
        # High push (instability, negative growth) + high pull (stability, high GDP)
        # But distance reduces it
    
    def test_update_geometries_from_countries(self):
        """Test updating geometries from country data."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        state.countries["USA"] = country
        
        # Initially area might be 0 or different
        initial_area = self.engine.geometries["USA"].area_km2
        
        self.engine.update_geometries_from_countries(state.countries)
        
        assert self.engine.geometries["USA"].area_km2 == 9833517


class TestBuildSpatialGraph:
    """Tests for build_spatial_graph function."""
    
    def test_build_graph(self):
        """Test building spatial graph."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        usa = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.8,
            trade_partners={"CAN": 5e11},
            diplomatic_relations={"CAN": 80, "MEX": 60},
        )
        can = Country(
            iso3="CAN", iso2="CA", name="Canada",
            region="Americas", subregion="Northern America",
            area_km2=9984670, population=38000000,
            gdp_usd=2e12, gdp_per_capita_usd=52000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.9,
            trade_partners={"USA": 4e11},
            diplomatic_relations={"USA": 80},
        )
        
        state.countries["USA"] = usa
        state.countries["CAN"] = can
        
        engine = SpatialEngine({})
        graph = build_spatial_graph(state, engine)
        
        # Check nodes
        assert graph.has_node("USA")
        assert graph.has_node("CAN")
        assert graph.nodes["USA"]["gdp"] == 25e12
        assert graph.nodes["USA"]["regime"] == "democracy"
        assert graph.nodes["USA"]["stability"] == 0.8
        
        # Check edges
        assert graph.has_edge("USA", "CAN")
        assert graph["USA"]["CAN"]["distance"] > 0
        assert graph["USA"]["CAN"]["type"] == "border"
        assert "trade_volume" in graph["USA"]["CAN"]
        # Last write wins - CAN's perspective (4e11) overwrites USA's (5e11)
        assert graph["USA"]["CAN"]["trade_volume"] == 4e11


class TestCountryGeometry:
    """Tests for CountryGeometry dataclass."""
    
    def test_creation(self):
        """Test creating a CountryGeometry."""
        geom = CountryGeometry(
            iso3="USA",
            centroid_lat=39.8,
            centroid_lon=-98.6,
            area_km2=9833517,
            neighbors=["CAN", "MEX"],
            coastline_km=19924,
            capital_lat=38.9,
            capital_lon=-77.0,
        )
        
        assert geom.iso3 == "USA"
        assert geom.centroid_lat == 39.8
        assert geom.centroid_lon == -98.6
        assert geom.area_km2 == 9833517
        assert geom.neighbors == ["CAN", "MEX"]
        assert geom.coastline_km == 19924
        assert geom.capital_lat == 38.9
        assert geom.capital_lon == -77.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])