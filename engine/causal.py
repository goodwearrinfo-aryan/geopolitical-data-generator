"""Causal engine for event propagation and downstream effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

import numpy as np

from schemas.core import (
    SimulationState, Country, PoliticalEvent, EventType, Conflict,
    ConflictIntensity, Sanction, Alliance, Treaty, Leader, RegimeType
)
from engine.temporal import TemporalEngine


@dataclass
class CausalLink:
    """Represents a causal relationship between events/state changes."""
    cause: str  # event type or state variable
    effect: str  # affected variable
    strength: float  # -1 to 1
    delay_timesteps: int = 0
    condition: Optional[Callable[[SimulationState, Country], bool]] = None
    description: str = ""


class CausalEngine:
    """Manages causal propagation of events through the system."""
    
    def __init__(self, rng: np.random.Generator, config: Dict[str, Any]):
        self.rng = rng
        self.config = config
        self.causal_graph: Dict[str, List[CausalLink]] = {}
        self._build_default_causal_graph()
    
    def _build_default_causal_graph(self) -> None:
        """Build default causal relationships."""
        # Coup effects
        self.add_link("coup", "stability_index", -0.3, 0, desc="Coup reduces stability")
        self.add_link("coup", "gdp_growth_rate", -0.05, 1, desc="Coup reduces growth")
        self.add_link("coup", "regime_type", 1.0, 0, 
                      condition=lambda s, c: True, desc="Successful coup changes regime")
        self.add_link("coup", "sanctions_on", 0.5, 1, desc="Coups trigger sanctions")
        self.add_link("coup", "military_expenditure_usd", 0.2, 1, desc="Coup increases military spending")
        
        # Protest effects
        self.add_link("protest", "stability_index", -0.05, 0, desc="Protests reduce stability")
        self.add_link("protest", "legitimacy", -0.02, 0, desc="Protests reduce legitimacy")
        self.add_link("protest", "policy_shift", 0.1, 2, desc="Protests may cause policy shifts")
        self.add_link("protest", "regime_change", 0.01, 4, 
                      condition=lambda s, c: c.stability_index < 0.3, desc="Large protests may topple regime")
        
        # Election effects
        self.add_link("election", "stability_index", 0.02, 0, desc="Elections slightly stabilize democracies")
        self.add_link("election", "legitimacy", 0.05, 0, desc="Elections boost legitimacy")
        self.add_link("election", "leader_change", 0.3, 0, desc="Elections may change leaders")
        self.add_link("election", "policy_shift", 0.2, 1, desc="New governments shift policy")
        
        # Regime change effects
        self.add_link("regime_change", "stability_index", -0.4, 0, desc="Regime change destabilizes")
        self.add_link("regime_change", "gdp_growth_rate", -0.08, 1, desc="Regime change hurts economy")
        self.add_link("regime_change", "sanctions_on", 0.4, 1, desc="Regime changes trigger sanctions")
        self.add_link("regime_change", "alliance_changes", 0.3, 2, desc="New regime shifts alliances")
        self.add_link("regime_change", "leader_change", 1.0, 0, desc="Regime change replaces leader")
        
        # Conflict effects
        self.add_link("conflict_start", "stability_index", -0.2, 0, desc="Conflict reduces stability")
        self.add_link("conflict_start", "gdp_growth_rate", -0.03, 1, desc="Conflict reduces growth")
        self.add_link("conflict_start", "military_expenditure_usd", 0.3, 1, desc="Conflict increases military spending")
        self.add_link("conflict_start", "refugees_origin", 10000, 1, desc="Conflict creates refugees")
        self.add_link("conflict_start", "trade_disruption", 0.2, 1, desc="Conflict disrupts trade")
        
        self.add_link("conflict_escalation", "battle_deaths", 1000, 0, desc="Escalation increases deaths")
        self.add_link("conflict_escalation", "nuclear_risk", 0.1, 0, desc="Escalation raises nuclear risk")
        self.add_link("conflict_escalation", "sanctions_on", 0.3, 1, desc="Escalation triggers sanctions")
        
        self.add_link("conflict_end", "stability_index", 0.1, 2, desc="Conflict end improves stability")
        self.add_link("conflict_end", "gdp_growth_rate", 0.02, 4, desc="Peace dividend")
        self.add_link("conflict_end", "refugees_return", 0.3, 4, desc="Refugees return after conflict")
        
        # Sanction effects
        self.add_link("sanction_imposed", "gdp_growth_rate", -0.02, 2, desc="Sanctions reduce growth")
        self.add_link("sanction_imposed", "trade_openness", -0.1, 1, desc="Sanctions reduce trade")
        self.add_link("sanction_imposed", "fx_reserves_usd", -0.15, 2, desc="Sanctions drain reserves")
        self.add_link("sanction_imposed", "inflation_rate", 0.05, 3, desc="Sanctions cause inflation")
        self.add_link("sanction_imposed", "stability_index", -0.05, 4, desc="Sanctions may destabilize")
        self.add_link("sanction_imposed", "regime_change", 0.02, 8, 
                      condition=lambda s, c: c.regime_type.value == "autocracy", desc="Sanctions may topple autocracies")
        
        self.add_link("sanction_lifted", "gdp_growth_rate", 0.01, 2, desc="Sanction relief boosts growth")
        self.add_link("sanction_lifted", "trade_openness", 0.05, 1, desc="Sanction relief restores trade")
        
        # Alliance effects
        self.add_link("alliance_formed", "military_cooperation", 0.5, 1, desc="Alliances increase military cooperation")
        self.add_link("alliance_formed", "trade_volume", 0.1, 2, desc="Alliances boost trade")
        self.add_link("alliance_formed", "diplomatic_support", 0.3, 0, desc="Alliances provide diplomatic backing")
        
        self.add_link("alliance_dissolved", "military_cooperation", -0.5, 1, desc="Broken alliances reduce cooperation")
        self.add_link("alliance_dissolved", "conflict_risk", 0.1, 2, desc="Broken alliances increase conflict risk")
        
        # Treaty effects
        self.add_link("treaty_signed", "diplomatic_relations", 0.2, 0, desc="Treaties improve relations")
        self.add_link("treaty_ratified", "compliance_cost", 0.05, 4, desc="Treaties impose compliance costs")
        self.add_link("treaty_ratified", "credibility", 0.1, 2, desc="Treaties enhance credibility")
        
        # Economic effects
        self.add_link("gdp_shock", "stability_index", -0.1, 1, desc="Economic shocks destabilize")
        self.add_link("gdp_shock", "protest", 0.2, 2, desc="Economic shocks trigger protests")
        self.add_link("gdp_shock", "leader_change", 0.1, 4, desc="Economic shocks threaten leaders")
        
        self.add_link("resource_discovery", "gdp_growth_rate", 0.02, 2, desc="Resource discoveries boost growth")
        self.add_link("resource_discovery", "corruption", 0.1, 4, desc="Resource wealth increases corruption")
        self.add_link("resource_discovery", "conflict_risk", 0.05, 2, desc="Resources attract conflict")
        
        # Leader effects
        self.add_link("leader_change", "policy_shift", 0.5, 1, desc="New leaders change policy")
        self.add_link("leader_change", "diplomatic_relations", 0.1, 2, desc="New leaders reset relations")
        
        # Demographic effects
        self.add_link("youth_bulge", "protest", 0.1, 0, desc="Youth bulge increases protest risk")
        self.add_link("youth_bulge", "conflict_risk", 0.05, 0, desc="Youth bulge increases conflict risk")
        self.add_link("urbanization", "gdp_growth_rate", 0.01, 4, desc="Urbanization boosts productivity")
        self.add_link("urbanization", "protest", 0.05, 2, desc="Urbanization facilitates protests")
        
        # Climate/Environmental
        self.add_link("climate_shock", "gdp_growth_rate", -0.02, 1, desc="Climate shocks reduce growth")
        self.add_link("climate_shock", "migration", 0.1, 2, desc="Climate shocks drive migration")
        self.add_link("climate_shock", "conflict_risk", 0.03, 4, desc="Climate stress increases conflict")
    
    def add_link(self, cause: str, effect: str, strength: float, delay: int = 0,
                 condition: Optional[Callable] = None, description: str = "") -> None:
        """Add a causal link to the graph."""
        if cause not in self.causal_graph:
            self.causal_graph[cause] = []
        
        self.causal_graph[cause].append(CausalLink(
            cause=cause,
            effect=effect,
            strength=strength,
            delay_timesteps=delay,
            condition=condition,
            description=description,
        ))
    
    def process_event(self, event: PoliticalEvent, state: SimulationState, 
                      temporal: TemporalEngine) -> List[tuple]:
        """Process an event and return downstream effects to apply."""
        effects = []
        cause_key = event.event_type.value
        
        if cause_key not in self.causal_graph:
            return effects
        
        country = state.countries.get(event.country_iso3)
        if not country:
            return effects
        
        for link in self.causal_graph[cause_key]:
            # Check condition
            if link.condition and not link.condition(state, country):
                continue
            
            # Calculate effect magnitude with noise
            magnitude = link.strength * self.rng.normal(1.0, 0.2)
            
            # Schedule for future timestep if delayed
            target_timestep = temporal.current_timestep + link.delay_timesteps
            
            effects.append({
                "target": event.country_iso3,
                "effect": link.effect,
                "magnitude": magnitude,
                "timestep": target_timestep,
                "source_event": event.id,
                "description": link.description,
            })
        
        return effects
    
    def apply_effects(self, effects: List[dict], state: SimulationState) -> None:
        """Apply scheduled effects to the simulation state."""
        for effect in effects:
            if effect["timestep"] != state.__dict__.get("_current_timestep", 0):
                continue  # Will be applied by temporal engine when timestep matches
            
            self._apply_single_effect(effect, state)
    
    def _apply_single_effect(self, effect: dict, state: SimulationState) -> None:
        """Apply a single effect to a country."""
        country = state.countries.get(effect["target"])
        if not country:
            return
        
        effect_name = effect["effect"]
        magnitude = effect["magnitude"]
        
        # Map effect names to country attributes
        effect_map = {
            "stability_index": lambda c, m: setattr(c, "stability_index", 
                max(0.0, min(1.0, c.stability_index + m))),
            "gdp_growth_rate": lambda c, m: setattr(c, "gdp_growth_rate", c.gdp_growth_rate + m),
            "regime_type": lambda c, m: setattr(c, "regime_type", 
                effect.get("new_regime_type", c.regime_type)) if m > 0.5 else None,
            "sanctions_on": lambda c, m: self._add_sanction(c, effect),
            "military_expenditure_usd": lambda c, m: setattr(c, "military_expenditure_usd", 
                c.military_expenditure_usd * (1 + m)),
            "refugees_origin": lambda c, m: setattr(c, "refugees_origin", c.refugees_origin + int(m)),
            "trade_disruption": lambda c, m: self._apply_trade_disruption(c, m, state),
            "battle_deaths": lambda c, m: None,  # Handled by conflict object
            "nuclear_risk": lambda c, m: None,  # Handled by conflict object
            "legitimacy": lambda c, m: None,  # Would need legitimacy attribute
            "policy_shift": lambda c, m: None,  # Triggers policy event
            "leader_change": lambda c, m: None,  # Triggers leader change event
            "alliance_changes": lambda c, m: None,  # Triggers alliance changes
            "trade_openness": lambda c, m: setattr(c, "trade_openness", 
                max(0.0, min(2.0, c.trade_openness + m))),
            "fx_reserves_usd": lambda c, m: setattr(c, "fx_reserves_usd", 
                max(0.0, c.fx_reserves_usd * (1 + m))),
            "inflation_rate": lambda c, m: setattr(c, "inflation_rate", c.inflation_rate + m),
            "conflict_risk": lambda c, m: None,  # Would update conflict probability
            "diplomatic_relations": lambda c, m: self._adjust_relations(c, m, state),
            "military_cooperation": lambda c, m: None,
            "trade_volume": lambda c, m: None,
            "diplomatic_support": lambda c, m: None,
            "compliance_cost": lambda c, m: None,
            "credibility": lambda c, m: None,
            "protest": lambda c, m: None,  # Triggers protest event
            "gdp_shock": lambda c, m: None,
            "resource_discovery": lambda c, m: None,
            "corruption": lambda c, m: None,
            "youth_bulge": lambda c, m: None,
            "urbanization": lambda c, m: None,
            "climate_shock": lambda c, m: None,
            "migration": lambda c, m: None,
        }
        
        if effect_name in effect_map:
            effect_map[effect_name](country, magnitude)
    
    def _add_sanction(self, country: Country, effect: dict) -> None:
        """Add a sanction to a country."""
        sanction_id = effect.get("sanction_id", "unknown")
        if sanction_id not in country.sanctions_on:
            country.sanctions_on.append(sanction_id)
    
    def _apply_trade_disruption(self, country: Country, magnitude: float, state: SimulationState) -> None:
        """Reduce trade with all partners."""
        for partner_iso3 in list(country.trade_partners.keys()):
            country.trade_partners[partner_iso3] *= (1 - magnitude * 0.5)
            # Also reduce partner's trade with this country
            partner = state.countries.get(partner_iso3)
            if partner and country.iso3 in partner.trade_partners:
                partner.trade_partners[country.iso3] *= (1 - magnitude * 0.5)
    
    def _adjust_relations(self, country: Country, magnitude: float, state: SimulationState) -> None:
        """Adjust diplomatic relations with all countries."""
        for target_iso3 in country.diplomatic_relations:
            current = country.diplomatic_relations[target_iso3]
            country.diplomatic_relations[target_iso3] = max(-100, min(100, current + magnitude * 10))


class ConflictDynamics:
    """Manages conflict escalation, de-escalation, and resolution."""
    
    def __init__(self, rng: np.random.Generator, config: Dict[str, Any]):
        self.rng = rng
        self.config = config
        
        # Escalation probabilities by intensity
        self.escalation_probs = {
            ConflictIntensity.TENSION: 0.15,
            ConflictIntensity.SKIRMISH: 0.25,
            ConflictIntensity.LOCALIZED: 0.20,
            ConflictIntensity.MAJOR: 0.15,
            ConflictIntensity.TOTAL: 0.05,  # Can't escalate further
        }
        
        self.deescalation_probs = {
            ConflictIntensity.TENSION: 0.30,
            ConflictIntensity.SKIRMISH: 0.20,
            ConflictIntensity.LOCALIZED: 0.15,
            ConflictIntensity.MAJOR: 0.10,
            ConflictIntensity.TOTAL: 0.05,
        }
        
        # Casualty rates by intensity (monthly)
        self.casualty_rates = {
            ConflictIntensity.TENSION: (0, 5),
            ConflictIntensity.SKIRMISH: (5, 50),
            ConflictIntensity.LOCALIZED: (50, 500),
            ConflictIntensity.MAJOR: (500, 5000),
            ConflictIntensity.TOTAL: (5000, 50000),
        }
    
    def step_conflicts(self, state: SimulationState, temporal: TemporalEngine) -> List[PoliticalEvent]:
        """Process all conflicts for one timestep."""
        events = []
        
        for conflict in list(state.conflicts.values()):
            if conflict.status != "ongoing":
                continue
            
            # Generate casualties
            min_cas, max_cas = self.casualty_rates[conflict.intensity]
            new_deaths = self.rng.integers(min_cas, max_cas + 1)
            conflict.battle_deaths += new_deaths
            
            # Civilian casualties (10-30% of battle deaths)
            civilian_deaths = int(new_deaths * self.rng.uniform(0.1, 0.3))
            conflict.civilian_deaths += civilian_deaths
            
            # Displacement
            displaced = int(new_deaths * self.rng.uniform(5, 20))
            conflict.displaced_persons += displaced
            
            # Check for escalation
            if self.rng.random() < self.escalation_probs[conflict.intensity]:
                new_intensity = self._escalate(conflict.intensity)
                if new_intensity != conflict.intensity:
                    conflict.intensity = new_intensity
                    events.append(PoliticalEvent(
                        country_iso3=conflict.primary_attacker,
                        event_type=EventType.CONFLICT_ESCALATION,
                        date=temporal.current_date,
                        description=f"Conflict {conflict.name} escalated to {new_intensity.value}",
                        actors=[conflict.primary_attacker, conflict.primary_defender],
                    ))
            
            # Check for de-escalation
            elif self.rng.random() < self.deescalation_probs[conflict.intensity]:
                new_intensity = self._deescalate(conflict.intensity)
                if new_intensity != conflict.intensity:
                    conflict.intensity = new_intensity
                    events.append(PoliticalEvent(
                        country_iso3=conflict.primary_attacker,
                        event_type=EventType.CONFLICT_DEESCALATION,
                        date=temporal.current_date,
                        description=f"Conflict {conflict.name} de-escalated to {new_intensity.value}",
                        actors=[conflict.primary_attacker, conflict.primary_defender],
                    ))
            
            # Check for ceasefire/peace
            if conflict.intensity == ConflictIntensity.TENSION:
                if self.rng.random() < 0.2:  # 20% chance of ceasefire at tension level
                    conflict.status = "ceasefire"
                    events.append(PoliticalEvent(
                        country_iso3=conflict.primary_attacker,
                        event_type=EventType.CEASEFIRE,
                        date=temporal.current_date,
                        description=f"Ceasefire in {conflict.name}",
                        actors=[conflict.primary_attacker, conflict.primary_defender] + conflict.mediator_countries,
                    ))
            
            # Check for conflict end
            if conflict.intensity == ConflictIntensity.TENSION and conflict.status == "ceasefire":
                if self.rng.random() < 0.15:  # 15% chance of peace treaty
                    conflict.status = "ended"
                    conflict.end_date = temporal.current_date
                    events.append(PoliticalEvent(
                        country_iso3=conflict.primary_attacker,
                        event_type=EventType.PEACE_TREATY,
                        date=temporal.current_date,
                        description=f"Peace treaty ending {conflict.name}",
                        actors=[conflict.primary_attacker, conflict.primary_defender] + conflict.mediator_countries,
                    ))
            
            # Economic cost estimation
            conflict.economic_cost_usd += new_deaths * 1_000_000 + displaced * 10_000
        
        return events
    
    def _escalate(self, current: ConflictIntensity) -> ConflictIntensity:
        """Escalate conflict intensity."""
        levels = list(ConflictIntensity)
        idx = levels.index(current)
        if idx < len(levels) - 1:
            return levels[idx + 1]
        return current
    
    def _deescalate(self, current: ConflictIntensity) -> ConflictIntensity:
        """De-escalate conflict intensity."""
        levels = list(ConflictIntensity)
        idx = levels.index(current)
        if idx > 0:
            return levels[idx - 1]
        return current
    
    def start_conflict(
        self,
        attacker: str,
        defender: str,
        conflict_type: str,
        state: SimulationState,
        temporal: TemporalEngine,
    ) -> Conflict:
        """Start a new conflict."""
        from schemas.core import ConflictType
        
        conflict = Conflict(
            name=f"{attacker}-{defender} Conflict {temporal.current_date.year}",
            conflict_type=ConflictType(conflict_type),
            intensity=ConflictIntensity.SKIRMISH,
            primary_attacker=attacker,
            primary_defender=defender,
            start_date=temporal.current_date,
            theater_countries=[attacker, defender],
        )
        
        state.conflicts[conflict.id] = conflict
        return conflict