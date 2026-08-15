"""Temporal engine for time-stepping simulation."""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from schemas.core import SimulationState, Country, PoliticalEvent, EventType, Leader


class TimestepFrequency(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass
class ScheduledEvent:
    """An event scheduled for a future timestep."""
    timestep: int
    callback: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0  # higher = earlier execution


class TemporalEngine:
    """Manages simulation time and event scheduling."""
    
    def __init__(
        self,
        start_date: date,
        end_date: date,
        frequency: TimestepFrequency = TimestepFrequency.MONTHLY,
        seed: int = 42,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.rng = np.random.default_rng(seed)
        
        self.current_timestep = 0
        self.current_date = start_date
        self.total_timesteps = self._calculate_total_timesteps()
        
        self.scheduled_events: List[ScheduledEvent] = []
        self.event_history: List[PoliticalEvent] = []
        
        # Timestep callbacks
        self.pre_timestep_hooks: List[Callable] = []
        self.post_timestep_hooks: List[Callable] = []
    
    def _calculate_total_timesteps(self) -> int:
        """Calculate number of timesteps between start and end date."""
        delta = self.end_date - self.start_date
        days = delta.days
        
        if self.frequency == TimestepFrequency.MONTHLY:
            return max(1, days // 30)
        elif self.frequency == TimestepFrequency.QUARTERLY:
            return max(1, days // 90)
        else:  # ANNUAL
            return max(1, days // 365)
    
    def advance(self) -> bool:
        """Advance to next timestep. Returns False if simulation ended."""
        if self.current_timestep >= self.total_timesteps:
            return False
        
        # Run pre-timestep hooks
        for hook in self.pre_timestep_hooks:
            hook(self)
        
        # Process scheduled events for this timestep
        self._process_scheduled_events()
        
        # Advance date
        self.current_timestep += 1
        self.current_date = self._calculate_date(self.current_timestep)
        
        # Run post-timestep hooks
        for hook in self.post_timestep_hooks:
            hook(self)
        
        return True
    
    def _calculate_date(self, timestep: int) -> date:
        """Calculate date for a given timestep."""
        if self.frequency == TimestepFrequency.MONTHLY:
            months = timestep
            year = self.start_date.year + months // 12
            month = (self.start_date.month - 1 + months % 12) + 1
            day = min(self.start_date.day, 28)  # Avoid month-end issues
            return date(year, month, day)
        elif self.frequency == TimestepFrequency.QUARTERLY:
            quarters = timestep
            year = self.start_date.year + quarters // 4
            quarter = (self.start_date.month - 1) // 3 + quarters % 4
            month = quarter * 3 + 1
            return date(year, month, 1)
        else:
            years = timestep
            return date(self.start_date.year + years, self.start_date.month, self.start_date.day)
    
    def _process_scheduled_events(self) -> None:
        """Process all events scheduled for current timestep."""
        due_events = [e for e in self.scheduled_events if e.timestep == self.current_timestep]
        due_events.sort(key=lambda e: -e.priority)
        
        for event in due_events:
            try:
                event.callback(*event.args, **event.kwargs)
            except Exception as e:
                print(f"Error in scheduled event: {e}")
        
        # Remove processed events
        self.scheduled_events = [e for e in self.scheduled_events if e.timestep != self.current_timestep]
    
    def schedule_event(
        self,
        timestep: int,
        callback: Callable,
        *args,
        priority: int = 0,
        **kwargs,
    ) -> None:
        """Schedule an event for a future timestep."""
        if timestep < self.current_timestep:
            raise ValueError(f"Cannot schedule event in past (timestep {timestep} < {self.current_timestep})")
        if timestep > self.total_timesteps:
            raise ValueError(f"Cannot schedule event beyond simulation end (timestep {timestep} > {self.total_timesteps})")
        
        self.scheduled_events.append(ScheduledEvent(
            timestep=timestep,
            callback=callback,
            args=args,
            kwargs=kwargs,
            priority=priority,
        ))
    
    def schedule_event_date(
        self,
        event_date: date,
        callback: Callable,
        *args,
        priority: int = 0,
        **kwargs,
    ) -> None:
        """Schedule an event for a specific date."""
        timestep = self._date_to_timestep(event_date)
        self.schedule_event(timestep, callback, *args, priority=priority, **kwargs)
    
    def _date_to_timestep(self, event_date: date) -> int:
        """Convert date to timestep index."""
        if event_date < self.start_date:
            return 0
        if event_date > self.end_date:
            return self.total_timesteps
        
        delta = event_date - self.start_date
        days = delta.days
        
        if self.frequency == TimestepFrequency.MONTHLY:
            return min(days // 30, self.total_timesteps)
        elif self.frequency == TimestepFrequency.QUARTERLY:
            return min(days // 90, self.total_timesteps)
        else:
            return min(days // 365, self.total_timesteps)
    
    def add_pre_hook(self, hook: Callable) -> None:
        self.pre_timestep_hooks.append(hook)
    
    def add_post_hook(self, hook: Callable) -> None:
        self.post_timestep_hooks.append(hook)
    
    def get_progress(self) -> float:
        """Get simulation progress as fraction 0-1."""
        return self.current_timestep / self.total_timesteps if self.total_timesteps > 0 else 1.0
    
    def remaining_timesteps(self) -> int:
        return max(0, self.total_timesteps - self.current_timestep)
    
    def record_event(self, event: PoliticalEvent) -> None:
        """Record a political event in history."""
        self.event_history.append(event)
    
    def get_events_in_range(self, start_timestep: int, end_timestep: int) -> List[PoliticalEvent]:
        """Get events within a timestep range."""
        return [
            e for e in self.event_history
            if start_timestep <= self._date_to_timestep(e.date) <= end_timestep
        ]


class EventGenerator:
    """Generates political events based on country state."""
    
    def __init__(self, rng: np.random.Generator, config: Dict[str, Any]):
        self.rng = rng
        self.config = config
        self.event_probabilities = self._load_probabilities()
    
    def _load_probabilities(self) -> Dict[EventType, Dict[str, float]]:
        """Load base event probabilities by regime type."""
        return {
            EventType.ELECTION: {
                "democracy": 0.25, "autocracy": 0.05, "anocracy": 0.15, "failed_state": 0.01
            },
            EventType.COUP: {
                "democracy": 0.005, "autocracy": 0.02, "anocracy": 0.05, "failed_state": 0.1
            },
            EventType.PROTEST: {
                "democracy": 0.1, "autocracy": 0.05, "anocracy": 0.2, "failed_state": 0.3
            },
            EventType.REGIME_CHANGE: {
                "democracy": 0.001, "autocracy": 0.01, "anocracy": 0.03, "failed_state": 0.05
            },
            EventType.LEADER_CHANGE: {
                "democracy": 0.15, "autocracy": 0.02, "anocracy": 0.1, "failed_state": 0.05
            },
            EventType.POLICY_SHIFT: {
                "democracy": 0.2, "autocracy": 0.1, "anocracy": 0.15, "failed_state": 0.05
            },
        }
    
    def generate_events(
        self,
        state: SimulationState,
        temporal: TemporalEngine,
    ) -> List[PoliticalEvent]:
        """Generate events for current timestep."""
        events = []
        
        for country in state.countries.values():
            country_events = self._generate_country_events(country, state, temporal)
            events.extend(country_events)
        
        return events
    
    def _generate_country_events(
        self,
        country: Country,
        state: SimulationState,
        temporal: TemporalEngine,
    ) -> List[PoliticalEvent]:
        events = []
        # Handle both enum and string regime_type
        regime_key = country.regime_type.value if hasattr(country.regime_type, 'value') else country.regime_type
        
        # Check for scheduled elections
        if country.next_election_year and temporal.current_date.year >= country.next_election_year:
            if self.rng.random() < 0.8:  # 80% chance election happens on schedule
                events.append(self._create_election_event(country, temporal.current_date))
                country.next_election_year += country.election_cycle_years
        
        # Random events based on probabilities
        for event_type, probs in self.event_probabilities.items():
            base_prob = probs.get(regime_key, 0.01)
            
            # Adjust based on country state
            adjusted_prob = base_prob * self._get_probability_multiplier(country, event_type)
            
            if self.rng.random() < adjusted_prob:
                event = self._create_event(country, event_type, temporal.current_date, state)
                if event:
                    events.append(event)
        
        return events
    
    def _get_probability_multiplier(self, country: Country, event_type: EventType) -> float:
        """Calculate probability multiplier based on country conditions."""
        multiplier = 1.0
        regime_key = country.regime_type.value if hasattr(country.regime_type, 'value') else country.regime_type
        
        if event_type == EventType.COUP:
            multiplier *= (1 + country.coup_risk * 10)
            multiplier *= (1 + (1 - country.stability_index) * 5)
        
        elif event_type == EventType.PROTEST:
            multiplier *= (1 + country.protest_level * 5)
            multiplier *= (1 + (1 - country.stability_index) * 3)
            if regime_key == "autocracy":
                multiplier *= 0.5  # Protests less likely but more significant
        
        elif event_type == EventType.REGIME_CHANGE:
            multiplier *= (1 - country.stability_index) * 10
            if country.leader_id:
                leader = None  # Would need to look up leader
                # multiplier *= (1 + leader.risk_tolerance) if leader else 1.0
        
        elif event_type == EventType.ELECTION:
            if regime_key == "democracy":
                multiplier *= 1.0
            else:
                multiplier *= 0.1
        
        return max(0.01, min(multiplier, 10.0))
    
    def _create_event(
        self,
        country: Country,
        event_type: EventType,
        event_date: date,
        state: SimulationState,
    ) -> Optional[PoliticalEvent]:
        """Create a specific event instance."""
        creators = {
            EventType.ELECTION: self._create_election_event,
            EventType.COUP: self._create_coup_event,
            EventType.PROTEST: self._create_protest_event,
            EventType.REGIME_CHANGE: self._create_regime_change_event,
            EventType.LEADER_CHANGE: self._create_leader_change_event,
            EventType.POLICY_SHIFT: self._create_policy_shift_event,
        }
        
        creator = creators.get(event_type)
        if creator:
            return creator(country, event_date)
        return None
    
    def _create_election_event(self, country: Country, event_date: date) -> PoliticalEvent:
        return PoliticalEvent(
            country_iso3=country.iso3,
            event_type=EventType.ELECTION,
            date=event_date,
            description=f"National election in {country.name}",
            actors=[country.ruling_party] if country.ruling_party else [],
            regime_change=False,
            stability_impact=self.rng.normal(0, 0.05),
            legitimacy_impact=self.rng.normal(0.02, 0.05),
            participants_estimate=int(country.population * self.rng.uniform(0.3, 0.7)),
            outcome=self.rng.choice(["incumbent_win", "opposition_win", "hung_parliament"], p=[0.6, 0.3, 0.1]),
        )
    
    def _create_coup_event(self, country: Country, event_date: date) -> PoliticalEvent:
        success = self.rng.random() < 0.3  # 30% success rate
        return PoliticalEvent(
            country_iso3=country.iso3,
            event_type=EventType.COUP,
            date=event_date,
            description=f"Coup attempt in {country.name} - {'successful' if success else 'failed'}",
            actors=["military"],
            regime_change=success,
            new_regime_type=RegimeType.AUTOCRACY if success else None,
            stability_impact=-0.3 if success else -0.1,
            legitimacy_impact=-0.4 if success else -0.1,
            casualties=int(self.rng.exponential(10)) if success else int(self.rng.exponential(2)),
            participants_estimate=int(self.rng.uniform(100, 10000)),
            outcome="success" if success else "failure",
        )
    
    def _create_protest_event(self, country: Country, event_date: date) -> PoliticalEvent:
        scale = self.rng.choice(["small", "medium", "large"], p=[0.5, 0.3, 0.2])
        size_mult = {"small": 0.001, "medium": 0.01, "large": 0.05}[scale]
        
        return PoliticalEvent(
            country_iso3=country.iso3,
            event_type=EventType.PROTEST,
            date=event_date,
            description=f"{scale.capitalize()} protests in {country.name}",
            actors=["civil_society", "opposition"],
            regime_change=False,
            stability_impact=-0.05 * {"small": 1, "medium": 3, "large": 10}[scale],
            legitimacy_impact=-0.02 * {"small": 1, "medium": 2, "large": 5}[scale],
            participants_estimate=int(country.population * size_mult * self.rng.uniform(0.5, 2)),
            duration_days=self.rng.integers(1, 30),
            outcome=self.rng.choice(["dispersed", "negotiated", "escalated", "ongoing"], p=[0.4, 0.3, 0.2, 0.1]),
        )
    
    def _create_regime_change_event(self, country: Country, event_date: date) -> PoliticalEvent:
        new_regime = self.rng.choice([r for r in RegimeType if r != country.regime_type])
        return PoliticalEvent(
            country_iso3=country.iso3,
            event_type=EventType.REGIME_CHANGE,
            date=event_date,
            description=f"Regime change in {country.name}: {country.regime_type.value} -> {new_regime.value}",
            actors=["military", "opposition", "foreign_powers"],
            regime_change=True,
            new_regime_type=new_regime,
            stability_impact=-0.5,
            legitimacy_impact=-0.3,
            casualties=int(self.rng.exponential(100)),
            participants_estimate=int(country.population * self.rng.uniform(0.01, 0.1)),
            outcome="completed",
        )
    
    def _create_leader_change_event(self, country: Country, event_date: date) -> PoliticalEvent:
        reasons = ["election", "resignation", "death", "term_limit", "coup", "impeachment"]
        reason = self.rng.choice(reasons, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
        
        return PoliticalEvent(
            country_iso3=country.iso3,
            event_type=EventType.LEADER_CHANGE,
            date=event_date,
            description=f"Leader change in {country.name} due to {reason}",
            actors=[country.ruling_party] if country.ruling_party else [],
            regime_change=reason == "coup",
            stability_impact=self.rng.normal(-0.05, 0.1),
            legitimacy_impact=self.rng.normal(0, 0.05),
            outcome=reason,
        )
    
    def _create_policy_shift_event(self, country: Country, event_date: date) -> PoliticalEvent:
        domains = ["economic", "foreign", "security", "social", "environmental"]
        domain = self.rng.choice(domains)
        
        return PoliticalEvent(
            country_iso3=country.iso3,
            event_type=EventType.POLICY_SHIFT,
            date=event_date,
            description=f"Major {domain} policy shift in {country.name}",
            actors=["government", "parliament"],
            stability_impact=self.rng.normal(0, 0.03),
            legitimacy_impact=self.rng.normal(0.01, 0.03),
            outcome=self.rng.choice(["implemented", "blocked", "modified", "delayed"], p=[0.5, 0.2, 0.2, 0.1]),
        )