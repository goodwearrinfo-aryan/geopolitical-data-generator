"""Unit tests for schemas.core module."""

from __future__ import annotations

import pytest
from datetime import date
from uuid import UUID

from schemas.core import (
    SimulationState, Country, Leader, PoliticalEvent, EventType,
    Conflict, ConflictType, ConflictIntensity, Alliance, AllianceType,
    Treaty, TreatyCategory, Sanction, SanctionType, RegimeType,
    CountryTier, ResourceType, EconomicIndicator, TradeFlow,
    MigrationFlow, DemographicProfile
)


class TestCountrySchema:
    """Tests for Country schema."""
    
    def test_valid_country_creation(self):
        """Test creating a valid country."""
        country = Country(
            iso3="USA",
            iso2="US",
            name="United States",
            region="Americas",
            subregion="Northern America",
            area_km2=9833517,
            population=331000000,
            gdp_usd=25_000_000_000_000,
            gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        
        assert country.iso3 == "USA"
        assert country.iso2 == "US"
        assert country.name == "United States"
        assert country.regime_type == RegimeType.DEMOCRACY
        assert country.population == 331000000
        assert country.gdp_usd == 25_000_000_000_000
    
    def test_regime_type_enum_coercion(self):
        """Test that string regime types are coerced to enum."""
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type="democracy",  # String input
        )
        
        assert country.regime_type == RegimeType.DEMOCRACY
    
    def test_invalid_regime_type_raises(self):
        """Test that invalid regime type raises validation error."""
        with pytest.raises(Exception):
            Country(
                iso3="USA", iso2="US", name="United States",
                region="Americas", subregion="Northern America",
                area_km2=9833517, population=331000000,
                gdp_usd=25e12, gdp_per_capita_usd=75000,
                regime_type="invalid_regime",
            )
    
    def test_serialization_roundtrip(self):
        """Test that country can be serialized and deserialized."""
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        
        data = country.model_dump()
        country2 = Country(**data)
        
        assert country == country2
        assert country.iso3 == country2.iso3
        assert country.regime_type == country2.regime_type
    
    def test_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        
        assert country.polity_score is None
        assert country.stability_index == 0.5
        assert country.coup_risk == 0.02
        assert country.protest_level == 0.1
        assert country.tier == CountryTier.CORE
        assert country.data_quality == 1.0


class TestLeaderSchema:
    """Tests for Leader schema."""
    
    def test_valid_leader_creation(self):
        """Test creating a valid leader."""
        leader = Leader(
            country_iso3="USA",
            name="Test President",
            title="President",
            birth_year=1960,
            gender="M",
            party="Democratic Party",
            ideology="center-left",
            start_date=date(2021, 1, 20),
            competence=0.8,
            charisma=0.7,
        )
        
        assert leader.country_iso3 == "USA"
        assert leader.name == "Test President"
        assert leader.is_active is True
        assert leader.competence == 0.8
    
    def test_gender_validation(self):
        """Test gender field validation."""
        leader = Leader(
            country_iso3="USA", name="Test", title="President",
            birth_year=1960, gender="F",
            start_date=date(2021, 1, 20),
        )
        assert leader.gender == "F"
        
        with pytest.raises(Exception):
            Leader(
                country_iso3="USA", name="Test", title="President",
                birth_year=1960, gender="X",
                start_date=date(2021, 1, 20),
            )
    
    def test_education_level_validation(self):
        """Test education level validation."""
        leader = Leader(
            country_iso3="USA", name="Test", title="President",
            birth_year=1960, gender="M",
            start_date=date(2021, 1, 20),
            education_level="postgraduate",
        )
        assert leader.education_level == "postgraduate"
        
        with pytest.raises(Exception):
            Leader(
                country_iso3="USA", name="Test", title="President",
                birth_year=1960, gender="M",
                start_date=date(2021, 1, 20),
                education_level="invalid",
            )


class TestPoliticalEventSchema:
    """Tests for PoliticalEvent schema."""
    
    def test_valid_event_creation(self):
        """Test creating a valid political event."""
        event = PoliticalEvent(
            country_iso3="USA",
            event_type=EventType.ELECTION,
            date=date(2024, 11, 5),
            description="Presidential election",
            actors=["Democratic Party", "Republican Party"],
            regime_change=False,
            stability_impact=0.02,
            legitimacy_impact=0.05,
            participants_estimate=150000000,
            outcome="democrat_win",
        )
        
        assert event.country_iso3 == "USA"
        assert event.event_type == EventType.ELECTION
        assert event.regime_change is False
        assert event.outcome == "democrat_win"
    
    def test_event_type_enum_coercion(self):
        """Test that string event types are coerced to enum."""
        event = PoliticalEvent(
            country_iso3="USA",
            event_type="coup",
            date=date(2023, 1, 1),
            description="Coup attempt",
            regime_change=True,
        )
        
        assert event.event_type == EventType.COUP
        assert event.regime_change is True


class TestConflictSchema:
    """Tests for Conflict schema."""
    
    def test_valid_conflict_creation(self):
        """Test creating a valid conflict."""
        conflict = Conflict(
            name="Test War",
            conflict_type=ConflictType.INTERSTATE,
            intensity=ConflictIntensity.MAJOR,
            primary_attacker="USA",
            primary_defender="IRN",
            start_date=date(2023, 1, 1),
            theater_countries=["USA", "IRN", "IRQ"],
            battle_deaths=10000,
            civilian_deaths=2000,
            status="ongoing",
        )
        
        assert conflict.name == "Test War"
        assert conflict.conflict_type == ConflictType.INTERSTATE
        assert conflict.intensity == ConflictIntensity.MAJOR
        assert conflict.primary_attacker == "USA"
        assert conflict.status == "ongoing"
    
    def test_conflict_type_enum_coercion(self):
        """Test that string conflict types are coerced to enum."""
        conflict = Conflict(
            name="Civil War",
            conflict_type="civil",
            intensity="localized",
            primary_attacker="SYR",
            primary_defender="SYR",
            start_date=date(2011, 3, 1),
            theater_countries=["SYR"],
        )
        
        assert conflict.conflict_type == ConflictType.CIVIL
        assert conflict.intensity == ConflictIntensity.LOCALIZED


class TestAllianceSchema:
    """Tests for Alliance schema."""
    
    def test_valid_alliance_creation(self):
        """Test creating a valid alliance."""
        alliance = Alliance(
            name="NATO",
            alliance_type=AllianceType.DEFENSE,
            members=["USA", "GBR", "FRA", "DEU", "CAN"],
            founding_date=date(1949, 4, 4),
            mutual_defense=True,
            intelligence_sharing=True,
            cohesion=0.85,
        )
        
        assert alliance.name == "NATO"
        assert alliance.alliance_type == AllianceType.DEFENSE
        assert len(alliance.members) == 5
        assert alliance.mutual_defense is True
        assert alliance.cohesion == 0.85


class TestTreatySchema:
    """Tests for Treaty schema."""
    
    def test_valid_treaty_creation(self):
        """Test creating a valid treaty."""
        treaty = Treaty(
            name="Paris Agreement",
            category=TreatyCategory.ENVIRONMENT,
            signatories=["USA", "CHN", "IND", "GBR", "FRA"],
            signed_date=date(2015, 12, 12),
            ratified_date=date(2016, 11, 4),
            entered_force_date=date(2016, 11, 4),
            is_active=True,
        )
        
        assert treaty.name == "Paris Agreement"
        assert treaty.category == TreatyCategory.ENVIRONMENT
        assert len(treaty.signatories) == 5
        assert treaty.is_active is True


class TestSanctionSchema:
    """Tests for Sanction schema."""
    
    def test_valid_sanction_creation(self):
        """Test creating a valid sanction."""
        sanction = Sanction(
            name="Iran Nuclear Sanctions",
            sanction_type=SanctionType.COMPREHENSIVE,
            target_country="IRN",
            imposing_countries=["USA", "GBR", "FRA", "DEU"],
            imposed_date=date(2018, 11, 5),
            status="active",
            financial_restrictions=True,
            trade_restrictions=True,
            arms_embargo=True,
        )
        
        assert sanction.name == "Iran Nuclear Sanctions"
        assert sanction.sanction_type == SanctionType.COMPREHENSIVE
        assert sanction.target_country == "IRN"
        assert sanction.financial_restrictions is True


class TestEconomicIndicatorSchema:
    """Tests for EconomicIndicator schema."""
    
    def test_valid_indicator_creation(self):
        """Test creating a valid economic indicator."""
        indicator = EconomicIndicator(
            country_iso3="USA",
            year=2023,
            quarter=4,
            gdp_nominal_usd=27_000_000_000_000,
            gdp_growth_pct=2.5,
            inflation_pct=3.2,
            unemployment_pct=3.7,
            debt_pct_gdp=120.0,
        )
        
        assert indicator.country_iso3 == "USA"
        assert indicator.year == 2023
        assert indicator.quarter == 4
        assert indicator.gdp_nominal_usd == 27_000_000_000_000


class TestSimulationState:
    """Tests for SimulationState container."""
    
    def test_empty_state(self):
        """Test creating an empty simulation state."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        assert state.timestep == 0
        assert state.date == date(2020, 1, 1)
        assert len(state.countries) == 0
        assert len(state.conflicts) == 0
        assert len(state.events) == 0
    
    def test_add_country(self):
        """Test adding a country to state."""
        state = SimulationState(timestep=1, date=date(2020, 2, 1))
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        
        state.countries[country.iso3] = country
        
        assert len(state.countries) == 1
        assert "USA" in state.countries
        assert state.countries["USA"].name == "United States"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])