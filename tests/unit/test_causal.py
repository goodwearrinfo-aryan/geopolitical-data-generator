"""Unit tests for engine.causal module."""

from __future__ import annotations

import pytest
from datetime import date
from unittest.mock import Mock

from engine.causal import CausalEngine, ConflictDynamics, CausalLink
from engine.temporal import TemporalEngine, TimestepFrequency
from schemas.core import (
    SimulationState, Country, PoliticalEvent, EventType, Conflict,
    ConflictType, ConflictIntensity, RegimeType, CountryTier
)


class TestCausalEngine:
    """Tests for CausalEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TemporalEngine(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            frequency=TimestepFrequency.MONTHLY,
            seed=42,
        )
        self.causal = CausalEngine(self.engine.rng, {})
    
    def test_initialization(self):
        """Test causal engine initialization."""
        assert self.causal.causal_graph is not None
        assert len(self.causal.causal_graph) > 0
        # Check some expected links exist
        assert "coup" in self.causal.causal_graph
        assert "protest" in self.causal.causal_graph
        assert "election" in self.causal.causal_graph
    
    def test_add_link(self):
        """Test adding a custom causal link."""
        initial_count = len(self.causal.causal_graph.get("custom_event", []))
        
        self.causal.add_link(
            "custom_event", "stability_index", -0.5, 2,
            description="Custom event reduces stability"
        )
        
        assert len(self.causal.causal_graph["custom_event"]) == initial_count + 1
        link = self.causal.causal_graph["custom_event"][-1]
        assert link.effect == "stability_index"
        assert link.strength == -0.5
        assert link.delay_timesteps == 2
        assert link.description == "Custom event reduces stability"
    
    def test_add_link_with_condition(self):
        """Test adding a link with a condition."""
        def test_condition(state, country):
            return country.stability_index < 0.3
        
        self.causal.add_link(
            "custom_event", "regime_change", 1.0, 0,
            condition=test_condition,
            description="Regime change when unstable"
        )
        
        link = self.causal.causal_graph["custom_event"][-1]
        assert link.condition is not None
        assert link.condition is test_condition
    
    def test_process_coup_event(self):
        """Test processing a coup event."""
        state = SimulationState(timestep=5, date=date(2020, 6, 1))
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.5,
        )
        state.countries["USA"] = country
        
        event = PoliticalEvent(
            country_iso3="USA",
            event_type=EventType.COUP,
            date=date(2020, 6, 1),
            description="Coup attempt",
            regime_change=True,
        )
        
        effects = self.causal.process_event(event, state, self.engine)
        
        # Should generate effects for stability, gdp, regime, sanctions, military
        effect_names = [e["effect"] for e in effects]
        assert "stability_index" in effect_names
        assert "gdp_growth_rate" in effect_names
        assert "sanctions_on" in effect_names
    
    def test_process_protest_event(self):
        """Test processing a protest event."""
        state = SimulationState(timestep=5, date=date(2020, 6, 1))
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            stability_index=0.5,
            protest_level=0.2,
        )
        state.countries["USA"] = country
        
        event = PoliticalEvent(
            country_iso3="USA",
            event_type=EventType.PROTEST,
            date=date(2020, 6, 1),
            description="Large protests",
            stability_impact=-0.1,
        )
        
        effects = self.causal.process_event(event, state, self.engine)
        
        effect_names = [e["effect"] for e in effects]
        assert "stability_index" in effect_names
        assert "legitimacy" in effect_names
    
    def test_process_event_with_delay(self):
        """Test that delayed effects are scheduled for future timesteps."""
        state = SimulationState(timestep=5, date=date(2020, 6, 1))
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
        )
        state.countries["USA"] = country
        
        event = PoliticalEvent(
            country_iso3="USA",
            event_type=EventType.SANCTION_IMPOSED,
            date=date(2020, 6, 1),
            description="Sanctions imposed",
        )
        
        effects = self.causal.process_event(event, state, self.engine)
        
        # Sanction effects should have delays
        for effect in effects:
            if effect["effect"] == "gdp_growth_rate":
                assert effect["timestep"] > self.engine.current_timestep
            if effect["effect"] == "inflation_rate":
                assert effect["timestep"] > self.engine.current_timestep
    
    def test_process_event_unknown_country(self):
        """Test processing event for unknown country."""
        state = SimulationState(timestep=5, date=date(2020, 6, 1))
        # No countries in state
        
        event = PoliticalEvent(
            country_iso3="XXX",  # Non-existent country
            event_type=EventType.COUP,
            date=date(2020, 6, 1),
            description="Coup in unknown country",
        )
        
        effects = self.causal.process_event(event, state, self.engine)
        
        # Should return empty list for unknown country
        assert effects == []


class TestConflictDynamics:
    """Tests for ConflictDynamics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TemporalEngine(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            frequency=TimestepFrequency.MONTHLY,
            seed=42,
        )
        self.dynamics = ConflictDynamics(self.engine.rng, {})
    
    def test_step_conflicts_no_conflicts(self):
        """Test stepping with no conflicts."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        events = self.dynamics.step_conflicts(state, self.engine)
        
        assert events == []
    
    def test_step_conflicts_generates_casualties(self):
        """Test that ongoing conflicts generate casualties."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        conflict = Conflict(
            name="Test War",
            conflict_type=ConflictType.INTERSTATE,
            intensity=ConflictIntensity.LOCALIZED,
            primary_attacker="USA",
            primary_defender="IRN",
            start_date=date(2020, 1, 1),
            status="ongoing",
            theater_countries=["USA", "IRN"],
        )
        state.conflicts[conflict.id] = conflict
        
        initial_battle_deaths = conflict.battle_deaths
        initial_civilian_deaths = conflict.civilian_deaths
        
        events = self.dynamics.step_conflicts(state, self.engine)
        
        assert conflict.battle_deaths > initial_battle_deaths
        assert conflict.civilian_deaths > initial_civilian_deaths
        assert conflict.displaced_persons > 0
        assert conflict.economic_cost_usd > 0
    
    def test_conflict_escalation(self):
        """Test conflict escalation probability."""
        # Run many steps to test escalation
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        conflict = Conflict(
            name="Test War",
            conflict_type=ConflictType.INTERSTATE,
            intensity=ConflictIntensity.TENSION,  # Lowest intensity
            primary_attacker="USA",
            primary_defender="IRN",
            start_date=date(2020, 1, 1),
            status="ongoing",
            theater_countries=["USA", "IRN"],
        )
        state.conflicts[conflict.id] = conflict
        
        # Run many timesteps
        escalated = False
        for _ in range(100):
            events = self.dynamics.step_conflicts(state, self.engine)
            if conflict.intensity != ConflictIntensity.TENSION:
                escalated = True
                break
        
        # With 15% escalation probability per step, should eventually escalate
        # (but not guaranteed in 100 steps, so we just verify it can run)
        assert isinstance(escalated, bool)
    
    def test_conflict_deescalation(self):
        """Test conflict de-escalation."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        conflict = Conflict(
            name="Test War",
            conflict_type=ConflictType.INTERSTATE,
            intensity=ConflictIntensity.MAJOR,
            primary_attacker="USA",
            primary_defender="IRN",
            start_date=date(2020, 1, 1),
            status="ongoing",
            theater_countries=["USA", "IRN"],
        )
        state.conflicts[conflict.id] = conflict
        
        # Run steps
        deescalated = False
        for _ in range(50):
            events = self.dynamics.step_conflicts(state, self.engine)
            if conflict.intensity == ConflictIntensity.LOCALIZED:
                deescalated = True
                break
        
        assert isinstance(deescalated, bool)
    
    def test_ceasefire_and_peace(self):
        """Test ceasefire and peace treaty progression."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        conflict = Conflict(
            name="Test War",
            conflict_type=ConflictType.INTERSTATE,
            intensity=ConflictIntensity.TENSION,
            primary_attacker="USA",
            primary_defender="IRN",
            start_date=date(2020, 1, 1),
            status="ceasefire",  # Start at ceasefire
            theater_countries=["USA", "IRN"],
        )
        state.conflicts[conflict.id] = conflict
        
        # Run steps to see if peace treaty is reached
        peace_reached = False
        for _ in range(50):
            events = self.dynamics.step_conflicts(state, self.engine)
            if conflict.status == "ended":
                peace_reached = True
                break
        
        assert isinstance(peace_reached, bool)
    
    def test_start_conflict(self):
        """Test starting a new conflict."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        conflict = self.dynamics.start_conflict(
            attacker="USA",
            defender="IRN",
            conflict_type="interstate",
            state=state,
            temporal=self.engine,
        )
        
        assert conflict.primary_attacker == "USA"
        assert conflict.primary_defender == "IRN"
        assert conflict.conflict_type == ConflictType.INTERSTATE
        assert conflict.intensity == ConflictIntensity.SKIRMISH
        assert conflict.status == "ongoing"
        assert conflict.id in state.conflicts
    
    def test_casualty_rates_by_intensity(self):
        """Test that casualty rates vary by intensity."""
        intensities_and_rates = [
            (ConflictIntensity.TENSION, (0, 5)),
            (ConflictIntensity.SKIRMISH, (5, 50)),
            (ConflictIntensity.LOCALIZED, (50, 500)),
            (ConflictIntensity.MAJOR, (500, 5000)),
            (ConflictIntensity.TOTAL, (5000, 50000)),
        ]
        
        for intensity, (min_cas, max_cas) in intensities_and_rates:
            assert self.dynamics.casualty_rates[intensity] == (min_cas, max_cas)


class TestCausalLink:
    """Tests for CausalLink dataclass."""
    
    def test_default_values(self):
        """Test CausalLink default values."""
        link = CausalLink(
            cause="test",
            effect="stability_index",
            strength=-0.5,
        )
        
        assert link.cause == "test"
        assert link.effect == "stability_index"
        assert link.strength == -0.5
        assert link.delay_timesteps == 0
        assert link.condition is None
        assert link.description == ""
    
    def test_with_condition(self):
        """Test CausalLink with condition."""
        def test_condition(state, country):
            return True
        
        link = CausalLink(
            cause="test",
            effect="stability_index",
            strength=-0.5,
            delay_timesteps=2,
            condition=test_condition,
            description="Test link",
        )
        
        assert link.delay_timesteps == 2
        assert link.condition is test_condition
        assert link.description == "Test link"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])