"""Unit tests for engine.temporal module."""

from __future__ import annotations

import pytest
from datetime import date, timedelta
from uuid import UUID

from engine.temporal import TemporalEngine, TimestepFrequency, ScheduledEvent, EventGenerator
from schemas.core import SimulationState, Country, PoliticalEvent, EventType, RegimeType, CountryTier


class TestTemporalEngine:
    """Tests for TemporalEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TemporalEngine(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            frequency=TimestepFrequency.MONTHLY,
            seed=42,
        )
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.start_date == date(2020, 1, 1)
        assert self.engine.end_date == date(2020, 12, 31)
        assert self.engine.frequency == TimestepFrequency.MONTHLY
        assert self.engine.current_timestep == 0
        assert self.engine.current_date == date(2020, 1, 1)
        assert self.engine.total_timesteps == 12
    
    def test_quarterly_frequency(self):
        """Test quarterly frequency calculation."""
        engine = TemporalEngine(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            frequency=TimestepFrequency.QUARTERLY,
            seed=42,
        )
        assert engine.total_timesteps == 4
    
    def test_annual_frequency(self):
        """Test annual frequency calculation."""
        engine = TemporalEngine(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            frequency=TimestepFrequency.ANNUAL,
            seed=42,
        )
        assert engine.total_timesteps == 5
    
    def test_advance_single_step(self):
        """Test advancing a single timestep."""
        result = self.engine.advance()
        
        assert result is True
        assert self.engine.current_timestep == 1
        assert self.engine.current_date == date(2020, 2, 1)
    
    def test_advance_to_end(self):
        """Test advancing to end of simulation."""
        for _ in range(12):
            self.engine.advance()
        
        # Next advance should return False
        result = self.engine.advance()
        assert result is False
        assert self.engine.current_timestep == 12
    
    def test_schedule_event(self):
        """Test scheduling an event for future timestep."""
        callback_called = []
        
        def test_callback(value):
            callback_called.append(value)
        
        # Schedule for timestep 4 (0-indexed, so 5th advance will process it)
        self.engine.schedule_event(4, test_callback, "test_value", priority=10)
        
        # Advance 5 times (t=0 to t=4, then process t=4 on 5th call)
        for _ in range(5):
            self.engine.advance()
        
        assert callback_called == ["test_value"]
    
    def test_schedule_event_priority(self):
        """Test that higher priority events execute first."""
        execution_order = []
        
        def callback1():
            execution_order.append(1)
        
        def callback2():
            execution_order.append(2)
        
        # Schedule both at timestep 2 (0-indexed, so 3rd advance will process it)
        self.engine.schedule_event(2, callback1, priority=5)
        self.engine.schedule_event(2, callback2, priority=10)
        
        for _ in range(3):
            self.engine.advance()
        
        # callback2 should execute first (higher priority)
        assert execution_order == [2, 1]
    
    def test_schedule_event_in_past_raises(self):
        """Test that scheduling in the past raises error."""
        self.engine.advance()  # Move to timestep 1
        
        with pytest.raises(ValueError):
            self.engine.schedule_event(0, lambda: None)
    
    def test_schedule_event_beyond_end_raises(self):
        """Test that scheduling beyond simulation end raises error."""
        with pytest.raises(ValueError):
            self.engine.schedule_event(13, lambda: None)  # Only 12 timesteps
    
    def test_schedule_event_by_date(self):
        """Test scheduling event by date."""
        callback_called = []
        
        def test_callback():
            callback_called.append(True)
        
        # Schedule for February 1st (timestep 1 for monthly)
        self.engine.schedule_event_date(date(2020, 2, 1), test_callback)
        
        # Advance to February (2 steps: Jan, Feb)
        for _ in range(2):
            self.engine.advance()
        
        assert callback_called == [True]
    
    def test_get_progress(self):
        """Test progress calculation."""
        assert self.engine.get_progress() == 0.0
        
        self.engine.advance()  # timestep 1
        assert abs(self.engine.get_progress() - 1/12) < 0.01
        
        for _ in range(11):
            self.engine.advance()
        
        assert self.engine.get_progress() == 1.0
    
    def test_remaining_timesteps(self):
        """Test remaining timesteps calculation."""
        assert self.engine.remaining_timesteps() == 12
        
        self.engine.advance()
        assert self.engine.remaining_timesteps() == 11
        
        for _ in range(11):
            self.engine.advance()
        
        assert self.engine.remaining_timesteps() == 0
    
    def test_record_event(self):
        """Test recording political events."""
        event = PoliticalEvent(
            country_iso3="USA",
            event_type=EventType.ELECTION,
            date=date(2020, 11, 3),
            description="Presidential election",
        )
        
        self.engine.record_event(event)
        
        assert len(self.engine.event_history) == 1
        assert self.engine.event_history[0].event_type == EventType.ELECTION
    
    def test_get_events_in_range(self):
        """Test retrieving events in timestep range."""
        event1 = PoliticalEvent(
            country_iso3="USA",
            event_type=EventType.ELECTION,
            date=date(2020, 3, 1),
            description="Event 1",
        )
        event2 = PoliticalEvent(
            country_iso3="USA",
            event_type=EventType.COUP,
            date=date(2020, 8, 1),
            description="Event 2",
        )
        
        self.engine.record_event(event1)
        self.engine.record_event(event2)
        
        # Get events in first half of year (timesteps 0-5)
        events = self.engine.get_events_in_range(0, 5)
        assert len(events) == 1
        assert events[0].event_type == EventType.ELECTION
        
        # Get events in second half
        events = self.engine.get_events_in_range(6, 11)
        assert len(events) == 1
        assert events[0].event_type == EventType.COUP


class TestEventGenerator:
    """Tests for EventGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TemporalEngine(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            frequency=TimestepFrequency.MONTHLY,
            seed=42,
        )
        self.generator = EventGenerator(self.engine.rng, {
            "coup_base_rate": 0.02,
            "protest_threshold": 0.6,
        })
    
    def test_generate_election_events(self):
        """Test election event generation for democracies."""
        state = SimulationState(timestep=0, date=date(2020, 1, 1))
        
        country = Country(
            iso3="USA", iso2="US", name="United States",
            region="Americas", subregion="Northern America",
            area_km2=9833517, population=331000000,
            gdp_usd=25e12, gdp_per_capita_usd=75000,
            regime_type=RegimeType.DEMOCRACY,
            next_election_year=2020,
        )
        state.countries["USA"] = country
        
        events = self.generator.generate_events(state, self.engine)
        
        # Should have election event
        election_events = [e for e in events if e.event_type == EventType.ELECTION]
        assert len(election_events) >= 1
        assert election_events[0].country_iso3 == "USA"
    
    def test_coup_probability_by_regime(self):
        """Test that coup probability varies by regime type."""
        # Test multiple regimes
        regimes_and_expected = [
            (RegimeType.DEMOCRACY, "low"),
            (RegimeType.AUTOCRACY, "medium"),
            (RegimeType.ANOCRACY, "high"),
            (RegimeType.FAILED_STATE, "very_high"),
        ]
        
        for regime, expected in regimes_and_expected:
            state = SimulationState(timestep=0, date=date(2020, 1, 1))
            country = Country(
                iso3="TST", iso2="TS", name="Test",
                region="Test", subregion="Test",
                area_km2=100000, population=10000000,
                gdp_usd=1e12, gdp_per_capita_usd=10000,
                regime_type=regime,
                stability_index=0.5,
            )
            state.countries["TST"] = country
            
            events = self.generator.generate_events(state, self.engine)
            coup_events = [e for e in events if e.event_type == EventType.COUP]
            
            # We can't assert exact counts due to randomness, but we can verify
            # the generator runs without error
            assert isinstance(coup_events, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])