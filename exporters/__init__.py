"""Export engines for multiple output formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schemas.core import (
    SimulationState, Country, PoliticalEvent, EventType, Conflict,
    Alliance, Treaty, Sanction, Leader, EconomicIndicator,
    TradeFlow, MigrationFlow, DemographicProfile, RegimeType,
    ConflictType, ConflictIntensity, AllianceType, SanctionType,
    TreatyCategory, ResourceType, CountryTier
)


def _enum_value(val: Any) -> Any:
    """Extract value from enum or return as-is."""
    if isinstance(val, Enum):
        return val.value
    return val


class BaseExporter:
    """Base class for exporters."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(self, state: SimulationState) -> None:
        raise NotImplementedError


class ParquetExporter(BaseExporter):
    """Export simulation state to partitioned Parquet files."""
    
    def export(self, state: SimulationState) -> None:
        """Export all data to Parquet with partitioning."""
        
        # Countries - one file per timestep partition
        self._export_countries(state)
        
        # Events - partitioned by year/month
        self._export_events(state)
        
        # Conflicts
        self._export_conflicts(state)
        
        # Alliances
        self._export_alliances(state)
        
        # Treaties
        self._export_treaties(state)
        
        # Sanctions
        self._export_sanctions(state)
        
        # Economic indicators
        self._export_economic_indicators(state)
        
        # Trade flows
        self._export_trade_flows(state)
        
        # Migration flows
        self._export_migration_flows(state)
        
        # Demographic profiles
        self._export_demographic_profiles(state)
        
        # Leaders
        self._export_leaders(state)
        
        # Summary metrics
        self._export_summary(state)
    
    def _export_countries(self, state: SimulationState) -> None:
        """Export country data."""
        rows = []
        for country in state.countries.values():
            rows.append({
                "timestep": state.timestep,
                "date": state.date,
                "iso3": country.iso3,
                "iso2": country.iso2,
                "name": country.name,
                "region": country.region,
                "subregion": country.subregion,
                "area_km2": country.area_km2,
                "population": country.population,
                "gdp_usd": country.gdp_usd,
                "gdp_per_capita_usd": country.gdp_per_capita_usd,
                "regime_type": _enum_value(country.regime_type),
                "polity_score": country.polity_score,
                "stability_index": country.stability_index,
                "state_fragility_index": country.state_fragility_index,
                "coup_risk": country.coup_risk,
                "protest_level": country.protest_level,
                "gdp_growth_rate": country.gdp_growth_rate,
                "inflation_rate": country.inflation_rate,
                "unemployment_rate": country.unemployment_rate,
                "debt_to_gdp": country.debt_to_gdp,
                "trade_openness": country.trade_openness,
                "currency": country.currency,
                "fx_reserves_usd": country.fx_reserves_usd,
                "military_expenditure_usd": country.military_expenditure_usd,
                "military_personnel": country.military_personnel,
                "nuclear_arsenal": country.nuclear_arsenal,
                "urbanization_rate": country.urbanization_rate,
                "median_age": country.median_age,
                "life_expectancy": country.life_expectancy,
                "literacy_rate": country.literacy_rate,
                "ethnic_fragmentation": country.ethnic_fragmentation,
                "religious_fragmentation": country.religious_fragmentation,
                "refugees_hosted": country.refugees_hosted,
                "refugees_origin": country.refugees_origin,
                "idps": country.idps,
                "data_quality": country.data_quality,
                "tier": _enum_value(country.tier),
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                self.output_dir / "countries.parquet",
            )
    
    def _export_events(self, state: SimulationState) -> None:
        """Export political events."""
        rows = []
        for event in state.events:
            rows.append({
                "timestep": state.timestep,
                "date": event.date,
                "event_id": str(event.id),
                "country_iso3": event.country_iso3,
                "event_type": event.event_type.value,
                "description": event.description,
                "actors": "|".join(event.actors),
                "regime_change": event.regime_change,
                "new_regime_type": event.new_regime_type.value if event.new_regime_type else None,
                "stability_impact": event.stability_impact,
                "legitimacy_impact": event.legitimacy_impact,
                "casualties": event.casualties,
                "participants_estimate": event.participants_estimate,
                "duration_days": event.duration_days,
                "outcome": event.outcome,
                "source": event.source,
                "confidence": event.confidence,
                "tags": "|".join(event.tags),
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                self.output_dir / "events.parquet",
                partition_cols=["timestep"] if len(df) > 1000 else None,
            )
    
    def _export_conflicts(self, state: SimulationState) -> None:
        """Export conflicts."""
        rows = []
        for conflict in state.conflicts.values():
            rows.append({
                "timestep": state.timestep,
                "conflict_id": str(conflict.id),
                "name": conflict.name,
                "conflict_type": conflict.conflict_type.value,
                "intensity": conflict.intensity.value,
                "primary_attacker": conflict.primary_attacker,
                "primary_defender": conflict.primary_defender,
                "secondary_participants": str(conflict.secondary_participants),
                "start_date": conflict.start_date,
                "end_date": conflict.end_date,
                "status": conflict.status,
                "theater_countries": "|".join(conflict.theater_countries),
                "contested_territories": "|".join(conflict.contested_territories),
                "battle_deaths": conflict.battle_deaths,
                "civilian_deaths": conflict.civilian_deaths,
                "displaced_persons": conflict.displaced_persons,
                "infrastructure_damage_usd": conflict.infrastructure_damage_usd,
                "economic_cost_usd": conflict.economic_cost_usd,
                "escalation_probability": conflict.escalation_probability,
                "nuclear_risk": conflict.nuclear_risk,
                "chemical_weapons_used": conflict.chemical_weapons_used,
                "peace_treaty_id": str(conflict.peace_treaty_id) if conflict.peace_treaty_id else None,
                "mediator_countries": "|".join(conflict.mediator_countries),
                "un_mission": conflict.un_mission,
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.output_dir / "conflicts.parquet")
    
    def _export_alliances(self, state: SimulationState) -> None:
        """Export alliances."""
        rows = []
        for alliance in state.alliances.values():
            rows.append({
                "timestep": state.timestep,
                "alliance_id": str(alliance.id),
                "name": alliance.name,
                "alliance_type": alliance.alliance_type.value,
                "members": "|".join(alliance.members),
                "founding_date": alliance.founding_date,
                "treaty_id": str(alliance.treaty_id) if alliance.treaty_id else None,
                "mutual_defense": alliance.mutual_defense,
                "consultation_required": alliance.consultation_required,
                "joint_exercises": alliance.joint_exercises,
                "intelligence_sharing": alliance.intelligence_sharing,
                "economic_integration": alliance.economic_integration,
                "is_active": alliance.is_active,
                "cohesion": alliance.cohesion,
                "dissolved_date": alliance.dissolved_date,
                "dissolution_reason": alliance.dissolution_reason,
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.output_dir / "alliances.parquet")
    
    def _export_treaties(self, state: SimulationState) -> None:
        """Export treaties."""
        rows = []
        for treaty in state.treaties.values():
            rows.append({
                "timestep": state.timestep,
                "treaty_id": str(treaty.id),
                "name": treaty.name,
                "category": treaty.category.value,
                "signatories": "|".join(treaty.signatories),
                "ratifiers": "|".join(treaty.ratifiers),
                "signed_date": treaty.signed_date,
                "ratified_date": treaty.ratified_date,
                "entered_force_date": treaty.entered_force_date,
                "expiry_date": treaty.expiry_date,
                "articles": "|".join(treaty.articles),
                "verification_mechanism": treaty.verification_mechanism,
                "dispute_resolution": treaty.dispute_resolution,
                "is_active": treaty.is_active,
                "withdrawals": str(treaty.withdrawals),
                "violations": "|".join(treaty.violations),
                "alliance_id": str(treaty.alliance_id) if treaty.alliance_id else None,
                "sanction_id": str(treaty.sanction_id) if treaty.sanction_id else None,
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.output_dir / "treaties.parquet")
    
    def _export_sanctions(self, state: SimulationState) -> None:
        """Export sanctions."""
        rows = []
        for sanction in state.sanctions.values():
            rows.append({
                "timestep": state.timestep,
                "sanction_id": str(sanction.id),
                "name": sanction.name,
                "sanction_type": sanction.sanction_type.value,
                "target_country": sanction.target_country,
                "imposing_countries": "|".join(sanction.imposing_countries),
                "imposing_organizations": "|".join(sanction.imposing_organizations),
                "imposed_date": sanction.imposed_date,
                "lifted_date": sanction.lifted_date,
                "review_date": sanction.review_date,
                "status": sanction.status,
                "sectors": "|".join(sanction.sectors),
                "entities": "|".join(sanction.entities),
                "individuals": "|".join(sanction.individuals),
                "financial_restrictions": sanction.financial_restrictions,
                "trade_restrictions": sanction.trade_restrictions,
                "travel_bans": sanction.travel_bans,
                "arms_embargo": sanction.arms_embargo,
                "estimated_cost_target_usd": sanction.estimated_cost_target_usd,
                "estimated_cost_imposers_usd": sanction.estimated_cost_imposers_usd,
                "compliance_rate": sanction.compliance_rate,
                "legal_basis": sanction.legal_basis,
                "un_resolution": sanction.un_resolution,
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.output_dir / "sanctions.parquet")
    
    def _export_economic_indicators(self, state: SimulationState) -> None:
        """Export economic indicators."""
        rows = []
        for indicator in state.economic_indicators:
            rows.append({
                "timestep": state.timestep,
                "indicator_id": str(indicator.id),
                "country_iso3": indicator.country_iso3,
                "year": indicator.year,
                "quarter": indicator.quarter,
                "month": indicator.month,
                "gdp_nominal_usd": indicator.gdp_nominal_usd,
                "gdp_ppp_usd": indicator.gdp_ppp_usd,
                "gdp_growth_pct": indicator.gdp_growth_pct,
                "gdp_per_capita_usd": indicator.gdp_per_capita_usd,
                "consumption_pct_gdp": indicator.consumption_pct_gdp,
                "investment_pct_gdp": indicator.investment_pct_gdp,
                "government_pct_gdp": indicator.government_pct_gdp,
                "exports_pct_gdp": indicator.exports_pct_gdp,
                "imports_pct_gdp": indicator.imports_pct_gdp,
                "inflation_pct": indicator.inflation_pct,
                "cpi_index": indicator.cpi_index,
                "ppi_index": indicator.ppi_index,
                "exchange_rate_usd": indicator.exchange_rate_usd,
                "interest_rate_pct": indicator.interest_rate_pct,
                "money_supply_growth_pct": indicator.money_supply_growth_pct,
                "unemployment_pct": indicator.unemployment_pct,
                "labor_force_participation_pct": indicator.labor_force_participation_pct,
                "youth_unemployment_pct": indicator.youth_unemployment_pct,
                "revenue_pct_gdp": indicator.revenue_pct_gdp,
                "expenditure_pct_gdp": indicator.expenditure_pct_gdp,
                "deficit_pct_gdp": indicator.deficit_pct_gdp,
                "debt_pct_gdp": indicator.debt_pct_gdp,
                "trade_balance_usd": indicator.trade_balance_usd,
                "current_account_usd": indicator.current_account_usd,
                "fdi_inflow_usd": indicator.fdi_inflow_usd,
                "fdi_outflow_usd": indicator.fdi_outflow_usd,
                "reserves_usd": indicator.reserves_usd,
                "energy_consumption_per_capita": indicator.energy_consumption_per_capita,
                "co2_emissions_per_capita": indicator.co2_emissions_per_capita,
                "renewable_pct_energy": indicator.renewable_pct_energy,
                "gini_coefficient": indicator.gini_coefficient,
                "hdi": indicator.hdi,
                "poverty_rate": indicator.poverty_rate,
                "source": indicator.source,
                "confidence": indicator.confidence,
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                self.output_dir / "economic_indicators.parquet",
                partition_cols=["year"] if len(df) > 1000 else None,
            )
    
    def _export_trade_flows(self, state: SimulationState) -> None:
        """Export trade flows."""
        rows = []
        for flow in state.trade_flows:
            rows.append({
                "timestep": state.timestep,
                "flow_id": str(flow.id),
                "year": flow.year,
                "exporter_iso3": flow.exporter_iso3,
                "importer_iso3": flow.importer_iso3,
                "product_code": flow.product_code,
                "product_category": flow.product_category,
                "value_usd": flow.value_usd,
                "quantity": flow.quantity,
                "quantity_unit": flow.quantity_unit,
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                self.output_dir / "trade_flows.parquet",
                partition_cols=["year"] if len(df) > 1000 else None,
            )
    
    def _export_migration_flows(self, state: SimulationState) -> None:
        """Export migration flows."""
        rows = []
        for flow in state.migration_flows:
            rows.append({
                "timestep": state.timestep,
                "flow_id": str(flow.id),
                "year": flow.year,
                "origin_iso3": flow.origin_iso3,
                "destination_iso3": flow.destination_iso3,
                "migrants": flow.migrants,
                "flow_type": flow.flow_type,
                "demographic_breakdown": str(flow.demographic_breakdown),
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.output_dir / "migration_flows.parquet")
    
    def _export_demographic_profiles(self, state: SimulationState) -> None:
        """Export demographic profiles."""
        rows = []
        for profile in state.demographic_profiles:
            rows.append({
                "timestep": state.timestep,
                "profile_id": str(profile.id),
                "country_iso3": profile.country_iso3,
                "year": profile.year,
                "total_population": profile.total_population,
                "male_population": profile.male_population,
                "female_population": profile.female_population,
                "population_growth_rate": profile.population_growth_rate,
                "population_density": profile.population_density,
                "age_0_14_pct": profile.age_0_14_pct,
                "age_15_64_pct": profile.age_15_64_pct,
                "age_65_plus_pct": profile.age_65_plus_pct,
                "median_age": profile.median_age,
                "dependency_ratio": profile.dependency_ratio,
                "urban_population_pct": profile.urban_population_pct,
                "urbanization_rate": profile.urbanization_rate,
                "major_cities": str(profile.major_cities),
                "life_expectancy_total": profile.life_expectancy_total,
                "life_expectancy_male": profile.life_expectancy_male,
                "life_expectancy_female": profile.life_expectancy_female,
                "infant_mortality_per_1000": profile.infant_mortality_per_1000,
                "maternal_mortality_per_100k": profile.maternal_mortality_per_100k,
                "fertility_rate": profile.fertility_rate,
                "contraceptive_prevalence": profile.contraceptive_prevalence,
                "hiv_prevalence": profile.hiv_prevalence,
                "literacy_rate_total": profile.literacy_rate_total,
                "literacy_rate_male": profile.literacy_rate_male,
                "literacy_rate_female": profile.literacy_rate_female,
                "primary_enrollment_pct": profile.primary_enrollment_pct,
                "secondary_enrollment_pct": profile.secondary_enrollment_pct,
                "tertiary_enrollment_pct": profile.tertiary_enrollment_pct,
                "mean_years_schooling": profile.mean_years_schooling,
                "net_migration_rate": profile.net_migration_rate,
                "refugee_stock": profile.refugee_stock,
                "asylum_seekers_pending": profile.asylum_seekers_pending,
                "idp_stock": profile.idp_stock,
                "remittances_received_usd": profile.remittances_received_usd,
                "ethnic_groups": str(profile.ethnic_groups),
                "religious_groups": str(profile.religious_groups),
                "languages": str(profile.languages),
                "labor_force_total": profile.labor_force_total,
                "labor_force_female_pct": profile.labor_force_female_pct,
                "employment_by_sector": str(profile.employment_by_sector),
                "vulnerable_employment_pct": profile.vulnerable_employment_pct,
                "source": profile.source,
                "confidence": profile.confidence,
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.output_dir / "demographic_profiles.parquet")
    
    def _export_leaders(self, state: SimulationState) -> None:
        """Export leaders."""
        rows = []
        for leader in state.leaders.values():
            rows.append({
                "timestep": state.timestep,
                "leader_id": str(leader.id),
                "country_iso3": leader.country_iso3,
                "name": leader.name,
                "title": leader.title,
                "birth_year": leader.birth_year,
                "gender": leader.gender,
                "party": leader.party,
                "ideology": leader.ideology,
                "start_date": leader.start_date,
                "end_date": leader.end_date,
                "term_limit_years": leader.term_limit_years,
                "competence": leader.competence,
                "charisma": leader.charisma,
                "risk_tolerance": leader.risk_tolerance,
                "hawkishness": leader.hawkishness,
                "corruption": leader.corruption,
                "military_background": leader.military_background,
                "education_level": leader.education_level,
                "previous_roles": "|".join(leader.previous_roles),
                "is_active": leader.is_active,
                "exit_reason": leader.exit_reason,
            })
        
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, self.output_dir / "leaders.parquet")
    
    def _export_summary(self, state: SimulationState) -> None:
        """Export summary metrics."""
        summary = {
            "timestep": state.timestep,
            "date": state.date,
            "global_gdp_usd": state.global_gdp_usd,
            "global_population": state.global_population,
            "global_co2_emissions": state.global_co2_emissions,
            "active_conflicts": state.active_conflicts,
            "total_battle_deaths": state.total_battle_deaths,
            "total_displaced": state.total_displaced,
            "n_countries": len(state.countries),
            "n_leaders": len(state.leaders),
            "n_conflicts": len(state.conflicts),
            "n_alliances": len(state.alliances),
            "n_treaties": len(state.treaties),
            "n_sanctions": len(state.sanctions),
            "n_events": len(state.events),
        }
        
        df = pd.DataFrame([summary])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.output_dir / "summary.parquet")


class CSVExporter(BaseExporter):
    """Export simulation state to CSV files."""
    
    def export(self, state: SimulationState) -> None:
        """Export all data to CSV."""
        # Similar to Parquet but write CSV
        self._write_csv(state.countries.values(), "countries", 
                       lambda c: self._country_to_dict(state, c))
        self._write_csv(state.events, "events", 
                       lambda e: self._event_to_dict(state, e))
        self._write_csv(state.conflicts.values(), "conflicts",
                       lambda c: self._conflict_to_dict(state, c))
        self._write_csv(state.alliances.values(), "alliances",
                       lambda a: self._alliance_to_dict(state, a))
        self._write_csv(state.treaties.values(), "treaties",
                       lambda t: self._treaty_to_dict(state, t))
        self._write_csv(state.sanctions.values(), "sanctions",
                       lambda s: self._sanction_to_dict(state, s))
        self._write_csv(state.economic_indicators, "economic_indicators",
                       lambda i: self._indicator_to_dict(state, i))
        self._write_csv(state.trade_flows, "trade_flows",
                       lambda f: self._trade_flow_to_dict(state, f))
        self._write_csv(state.migration_flows, "migration_flows",
                       lambda f: self._migration_flow_to_dict(state, f))
        self._write_csv(state.demographic_profiles, "demographic_profiles",
                       lambda p: self._demographic_to_dict(state, p))
        self._write_csv(state.leaders.values(), "leaders",
                       lambda l: self._leader_to_dict(state, l))
    
    def _write_csv(self, items, filename: str, mapper: callable) -> None:
        rows = [mapper(item) for item in items]
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.output_dir / f"{filename}.csv", index=False)
    
    def _country_to_dict(self, state: SimulationState, c: Country) -> Dict:
        regime = c.regime_type
        if hasattr(regime, 'value'):
            regime = regime.value
        return {
            "timestep": state.timestep, "date": state.date, "iso3": c.iso3,
            "name": c.name, "regime_type": regime,
            "population": c.population, "gdp_usd": c.gdp_usd,
            "stability_index": c.stability_index, "gdp_growth_rate": c.gdp_growth_rate,
        }
    
    def _event_to_dict(self, state: SimulationState, e: PoliticalEvent) -> Dict:
        return {
            "timestep": state.timestep, "date": e.date, "event_id": str(e.id),
            "country_iso3": e.country_iso3, "event_type": e.event_type.value,
            "description": e.description, "regime_change": e.regime_change,
            "casualties": e.casualties, "outcome": e.outcome,
        }
    
    def _conflict_to_dict(self, state: SimulationState, c: Conflict) -> Dict:
        return {
            "timestep": state.timestep, "conflict_id": str(c.id),
            "name": c.name, "conflict_type": c.conflict_type.value,
            "intensity": c.intensity.value, "attacker": c.primary_attacker,
            "defender": c.primary_defender, "battle_deaths": c.battle_deaths,
            "civilian_deaths": c.civilian_deaths, "status": c.status,
        }
    
    def _alliance_to_dict(self, state: SimulationState, a: Alliance) -> Dict:
        return {
            "timestep": state.timestep, "alliance_id": str(a.id),
            "name": a.name, "type": a.alliance_type.value,
            "members": "|".join(a.members), "active": a.is_active,
        }
    
    def _treaty_to_dict(self, state: SimulationState, t: Treaty) -> Dict:
        return {
            "timestep": state.timestep, "treaty_id": str(t.id),
            "name": t.name, "category": t.category.value,
            "signatories": "|".join(t.signatories), "active": t.is_active,
        }
    
    def _sanction_to_dict(self, state: SimulationState, s: Sanction) -> Dict:
        return {
            "timestep": state.timestep, "sanction_id": str(s.id),
            "name": s.name, "type": s.sanction_type.value,
            "target": s.target_country, "imposers": "|".join(s.imposing_countries),
            "status": s.status,
        }
    
    def _indicator_to_dict(self, state: SimulationState, i: EconomicIndicator) -> Dict:
        return {
            "timestep": state.timestep, "indicator_id": str(i.id),
            "country_iso3": i.country_iso3, "year": i.year,
            "gdp_nominal_usd": i.gdp_nominal_usd, "gdp_growth_pct": i.gdp_growth_pct,
            "inflation_pct": i.inflation_pct, "unemployment_pct": i.unemployment_pct,
        }
    
    def _trade_flow_to_dict(self, state: SimulationState, f: TradeFlow) -> Dict:
        return {
            "timestep": state.timestep, "flow_id": str(f.id),
            "year": f.year, "exporter": f.exporter_iso3,
            "importer": f.importer_iso3, "product": f.product_code,
            "value_usd": f.value_usd,
        }
    
    def _migration_flow_to_dict(self, state: SimulationState, f: MigrationFlow) -> Dict:
        return {
            "timestep": state.timestep, "flow_id": str(f.id),
            "year": f.year, "origin": f.origin_iso3,
            "destination": f.destination_iso3, "migrants": f.migrants,
            "type": f.flow_type,
        }
    
    def _demographic_to_dict(self, state: SimulationState, p: DemographicProfile) -> Dict:
        return {
            "timestep": state.timestep, "profile_id": str(p.id),
            "country_iso3": p.country_iso3, "year": p.year,
            "population": p.total_population, "growth_rate": p.population_growth_rate,
            "urban_pct": p.urban_population_pct, "median_age": p.median_age,
        }
    
    def _leader_to_dict(self, state: SimulationState, l: Leader) -> Dict:
        return {
            "timestep": state.timestep, "leader_id": str(l.id),
            "country_iso3": l.country_iso3, "name": l.name,
            "title": l.title, "start_date": l.start_date,
            "active": l.is_active,
        }


class GeoJSONExporter(BaseExporter):
    """Export geographic data to GeoJSON for GIS visualization."""
    
    def export(self, state: SimulationState) -> None:
        """Export country geometries with attributes."""
        features = []
        
        for country in state.countries.values():
            # In production, load actual geometries from Natural Earth
            # For now, create point features at centroids
            feature = {
                "type": "Feature",
                "properties": {
                    "iso3": country.iso3,
                    "iso2": country.iso2,
                    "name": country.name,
                    "region": country.region,
                    "regime_type": _enum_value(country.regime_type),
                    "population": country.population,
                    "gdp_usd": country.gdp_usd,
                    "gdp_per_capita_usd": country.gdp_per_capita_usd,
                    "stability_index": country.stability_index,
                    "gdp_growth_rate": country.gdp_growth_rate,
                    "military_expenditure_usd": country.military_expenditure_usd,
                    "nuclear_arsenal": country.nuclear_arsenal,
                    "refugees_hosted": country.refugees_hosted,
                    "refugees_origin": country.refugees_origin,
                    "idps": country.idps,
                    "timestep": state.timestep,
                    "date": state.date.isoformat(),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [0, 0],  # Placeholder - would use actual centroid
                },
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        
        import json
        with open(self.output_dir / f"countries_timestep_{state.timestep}.geojson", "w") as f:
            json.dump(geojson, f)
        
        # Also export conflicts as lines/polygons
        self._export_conflicts_geojson(state)
    
    def _export_conflicts_geojson(self, state: SimulationState) -> None:
        """Export conflicts as GeoJSON features."""
        features = []
        
        for conflict in state.conflicts.values():
            # Create line between attacker and defender capitals
            feature = {
                "type": "Feature",
                "properties": {
                    "conflict_id": str(conflict.id),
                    "name": conflict.name,
                    "conflict_type": conflict.conflict_type.value,
                    "intensity": conflict.intensity.value,
                    "attacker": conflict.primary_attacker,
                    "defender": conflict.primary_defender,
                    "battle_deaths": conflict.battle_deaths,
                    "civilian_deaths": conflict.civilian_deaths,
                    "status": conflict.status,
                    "timestep": state.timestep,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [0, 0]],  # Placeholder
                },
            }
            features.append(feature)
        
        if features:
            geojson = {"type": "FeatureCollection", "features": features}
            import json
            with open(self.output_dir / f"conflicts_timestep_{state.timestep}.geojson", "w") as f:
                json.dump(geojson, f)


class NetworkExporter(BaseExporter):
    """Export relationship networks for graph analysis."""
    
    def export(self, state: SimulationState) -> None:
        """Export networks in various formats."""
        import networkx as nx
        import json
        
        # Diplomatic relations network
        G_diplo = nx.Graph()
        for iso3, country in state.countries.items():
            regime_val = _enum_value(country.regime_type)
            G_diplo.add_node(iso3, 
                           name=country.name,
                           regime=regime_val,
                           gdp=country.gdp_usd,
                           stability=country.stability_index)
        
        for iso3, country in state.countries.items():
            for target, level in country.diplomatic_relations.items():
                if target in state.countries and level != 0:
                    G_diplo.add_edge(iso3, target, weight=level, type="diplomatic")
        
        # Trade network
        G_trade = nx.DiGraph()
        for iso3, country in state.countries.items():
            G_trade.add_node(iso3)
        
        for iso3, country in state.countries.items():
            for partner, volume in country.trade_partners.items():
                if partner in state.countries and volume > 0:
                    G_trade.add_edge(iso3, partner, weight=volume, type="trade")
        
        # Alliance network
        G_alliance = nx.Graph()
        for iso3, country in state.countries.items():
            G_alliance.add_node(iso3)
        
        for alliance in state.alliances.values():
            if alliance.is_active:
                members = alliance.members
                for i, m1 in enumerate(members):
                    for m2 in members[i+1:]:
                        if m1 in state.countries and m2 in state.countries:
                            G_alliance.add_edge(m1, m2, 
                                              alliance=str(alliance.id),
                                              type=_enum_value(alliance.alliance_type))
        
        # Export as GraphML
        nx.write_graphml(G_diplo, self.output_dir / f"diplomatic_network_t{state.timestep}.graphml")
        nx.write_graphml(G_trade, self.output_dir / f"trade_network_t{state.timestep}.graphml")
        nx.write_graphml(G_alliance, self.output_dir / f"alliance_network_t{state.timestep}.graphml")
        
        # Export as node/link JSON for D3.js
        for name, G in [("diplomatic", G_diplo), ("trade", G_trade), ("alliance", G_alliance)]:
            data = nx.node_link_data(G)
            with open(self.output_dir / f"{name}_network_t{state.timestep}.json", "w") as f:
                json.dump(data, f)


class KafkaExporter(BaseExporter):
    """Stream simulation events to Kafka topics.
    
    Topics:
    - geopolitical.countries: Country state snapshots (keyed by iso3)
    - geopolitical.events: Political events (keyed by event_id)
    - geopolitical.conflicts: Conflict updates (keyed by conflict_id)
    - geopolitical.economic: Economic indicators (keyed by iso3)
    - geopolitical.trade: Trade flows (keyed by exporter_iso3)
    - geopolitical.migration: Migration flows (keyed by origin_iso3)
    """
    
    def __init__(
        self, 
        output_dir: str, 
        bootstrap_servers: str = "localhost:9092",
        topic_prefix: str = "geopolitical",
        flush_interval: int = 100,
    ):
        super().__init__(output_dir)
        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self.flush_interval = flush_interval
        self._producer = None
        self._message_count = 0
    
    def _get_producer(self):
        """Lazy-initialize Kafka producer."""
        if self._producer is None:
            try:
                from confluent_kafka import Producer
            except ImportError:
                raise ImportError("confluent-kafka required for KafkaExporter. Install with: pip install confluent-kafka")
            
            self._producer = Producer({
                'bootstrap.servers': self.bootstrap_servers,
                'acks': 'all',
                'retries': 3,
                'linger.ms': 10,
                'batch.size': 16384,
                'compression.type': 'snappy',
            })
        return self._producer
    
    def _serialize(self, obj: Any) -> bytes:
        """Serialize object to JSON bytes."""
        import json
        return json.dumps(obj, default=str).encode('utf-8')
    
    def _delivery_report(self, err, msg):
        """Callback for message delivery confirmation."""
        if err is not None:
            print(f"Kafka delivery failed: {err}")
    
    def export(self, state: SimulationState) -> None:
        """Export current simulation state to Kafka."""
        producer = self._get_producer()
        
        # Countries - current state snapshot
        for country in state.countries.values():
            producer.produce(
                f"{self.topic_prefix}.countries",
                key=country.iso3.encode(),
                value=self._serialize(self._country_to_dict(state, country)),
                callback=self._delivery_report,
            )
        
        # Events - only new events this timestep
        for event in state.events:
            producer.produce(
                f"{self.topic_prefix}.events",
                key=str(event.id).encode(),
                value=self._serialize(self._event_to_dict(state, event)),
                callback=self._delivery_report,
            )
        
        # Conflicts
        for conflict in state.conflicts.values():
            producer.produce(
                f"{self.topic_prefix}.conflicts",
                key=str(conflict.id).encode(),
                value=self._serialize(self._conflict_to_dict(state, conflict)),
                callback=self._delivery_report,
            )
        
        # Economic indicators
        for indicator in state.economic_indicators:
            if indicator.year == state.date.year:
                producer.produce(
                    f"{self.topic_prefix}.economic",
                    key=indicator.country_iso3.encode(),
                    value=self._serialize(self._indicator_to_dict(state, indicator)),
                    callback=self._delivery_report,
                )
        
        # Trade flows
        for flow in state.trade_flows:
            if flow.year == state.date.year:
                producer.produce(
                    f"{self.topic_prefix}.trade",
                    key=flow.exporter_iso3.encode(),
                    value=self._serialize(self._trade_flow_to_dict(state, flow)),
                    callback=self._delivery_report,
                )
        
        # Migration flows
        for flow in state.migration_flows:
            if flow.year == state.date.year:
                producer.produce(
                    f"{self.topic_prefix}.migration",
                    key=flow.origin_iso3.encode(),
                    value=self._serialize(self._migration_flow_to_dict(state, flow)),
                    callback=self._delivery_report,
                )
        
        # Flush periodically
        self._message_count += 1
        if self._message_count % self.flush_interval == 0:
            producer.flush(timeout=5.0)
    
    # Dict conversion methods (shared with CSVExporter)
    def _country_to_dict(self, state: SimulationState, c: Country) -> Dict:
        regime = c.regime_type
        if hasattr(regime, 'value'):
            regime = regime.value
        return {
            "timestep": state.timestep, "date": state.date, "iso3": c.iso3,
            "name": c.name, "regime_type": regime,
            "population": c.population, "gdp_usd": c.gdp_usd,
            "stability_index": c.stability_index, "gdp_growth_rate": c.gdp_growth_rate,
        }
    
    def _event_to_dict(self, state: SimulationState, e: PoliticalEvent) -> Dict:
        return {
            "timestep": state.timestep, "date": e.date, "event_id": str(e.id),
            "country_iso3": e.country_iso3, "event_type": _enum_value(e.event_type),
            "description": e.description, "regime_change": e.regime_change,
            "casualties": e.casualties, "outcome": e.outcome,
        }
    
    def _conflict_to_dict(self, state: SimulationState, c: Conflict) -> Dict:
        return {
            "timestep": state.timestep, "conflict_id": str(c.id),
            "name": c.name, "conflict_type": _enum_value(c.conflict_type),
            "intensity": _enum_value(c.intensity), "attacker": c.primary_attacker,
            "defender": c.primary_defender, "battle_deaths": c.battle_deaths,
            "civilian_deaths": c.civilian_deaths, "status": c.status,
        }
    
    def _alliance_to_dict(self, state: SimulationState, a: Alliance) -> Dict:
        return {
            "timestep": state.timestep, "alliance_id": str(a.id),
            "name": a.name, "type": _enum_value(a.alliance_type),
            "members": "|".join(a.members), "active": a.is_active,
        }
    
    def _treaty_to_dict(self, state: SimulationState, t: Treaty) -> Dict:
        return {
            "timestep": state.timestep, "treaty_id": str(t.id),
            "name": t.name, "category": _enum_value(t.category),
            "signatories": "|".join(t.signatories), "active": t.is_active,
        }
    
    def _sanction_to_dict(self, state: SimulationState, s: Sanction) -> Dict:
        return {
            "timestep": state.timestep, "sanction_id": str(s.id),
            "name": s.name, "type": _enum_value(s.sanction_type),
            "target": s.target_country, "imposers": "|".join(s.imposing_countries),
            "status": s.status,
        }
    
    def _indicator_to_dict(self, state: SimulationState, i: EconomicIndicator) -> Dict:
        return {
            "timestep": state.timestep, "indicator_id": str(i.id),
            "country_iso3": i.country_iso3, "year": i.year,
            "gdp_nominal_usd": i.gdp_nominal_usd, "gdp_growth_pct": i.gdp_growth_pct,
            "inflation_pct": i.inflation_pct, "unemployment_pct": i.unemployment_pct,
        }
    
    def _trade_flow_to_dict(self, state: SimulationState, f: TradeFlow) -> Dict:
        return {
            "timestep": state.timestep, "flow_id": str(f.id),
            "year": f.year, "exporter": f.exporter_iso3,
            "importer": f.importer_iso3, "product": f.product_code,
            "value_usd": f.value_usd,
        }
    
    def _migration_flow_to_dict(self, state: SimulationState, f: MigrationFlow) -> Dict:
        return {
            "timestep": state.timestep, "flow_id": str(f.id),
            "year": f.year, "origin": f.origin_iso3,
            "destination": f.destination_iso3, "migrants": f.migrants,
            "type": f.flow_type,
        }

    def close(self):
        """Close producer and flush."""
        self.flush()
        if self._producer:
            self._producer = None


class Neo4jExporter(BaseExporter):
    """Export simulation state as a property graph to Neo4j.
    
    Nodes:
    - Country: iso3, name, regime_type, region, gdp_usd, population, stability_index
    - Leader: id, name, country_iso3, title, ideology, start_date
    - Conflict: id, name, type, intensity, battle_deaths, status
    - Alliance: id, name, type, founding_date, cohesion
    - Treaty: id, name, category, signed_date, is_active
    - Sanction: id, name, type, target_country, status
    
    Relationships:
    - (Country)-[:BORDERS {length_km}]->(Country)
    - (Country)-[:TRADES_WITH {value_usd, year}]->(Country)
    - (Country)-[:ALLIED_WITH {type, since}]->(Country)
    - (Country)-[:HAS_TREATY {category}]->(Treaty)
    - (Country)-[:IMPOSES_SANCTION {type, since}]->(Sanction)
    - (Country)-[:HAS_LEADER {since}]->(Leader)
    - (Conflict)-[:INVOLVES {role}]->(Country)
    - (Alliance)-[:HAS_MEMBER]->(Country)
    """
    
    def __init__(
        self, 
        output_dir: str, 
        uri: str = "bolt://localhost:7687", 
        user: str = "neo4j", 
        password: str = "password",
        batch_size: int = 1000,
    ):
        super().__init__(output_dir)
        self.uri = uri
        self.user = user
        self.password = password
        self.batch_size = batch_size
        self._driver = None
    
    def _get_driver(self):
        """Lazy-initialize Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError:
                raise ImportError("neo4j driver required for Neo4jExporter. Install with: pip install neo4j")
            
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver
    
    def export(self, state: SimulationState) -> None:
        """Export current simulation state to Neo4j."""
        driver = self._get_driver()
        
        with driver.session() as session:
            # Upsert countries
            for country in state.countries.values():
                session.execute_write(self._upsert_country, country, state.timestep)
            
            # Trade relationships
            for country in state.countries.values():
                for partner, volume in country.trade_partners.items():
                    if partner in state.countries and volume > 0:
                        session.execute_write(self._upsert_trade, country.iso3, partner, volume, state.date.year)
            
            # Diplomatic relations
            for country in state.countries.values():
                for target, level in country.diplomatic_relations.items():
                    if target in state.countries and level != 0:
                        session.execute_write(self._upsert_diplomatic, country.iso3, target, level)
            
            # Alliances
            for alliance in state.alliances.values():
                session.execute_write(self._upsert_alliance, alliance, state.timestep)
            
            # Conflicts
            for conflict in state.conflicts.values():
                session.execute_write(self._upsert_conflict, conflict, state.timestep)
            
            # Treaties
            for treaty in state.treaties.values():
                session.execute_write(self._upsert_treaty, treaty)
            
            # Sanctions
            for sanction in state.sanctions.values():
                session.execute_write(self._upsert_sanction, sanction)
            
            # Leaders
            for leader in state.leaders.values():
                if leader.is_active:
                    session.execute_write(self._upsert_leader, leader)
    
    @staticmethod
    def _upsert_country(tx, country, timestep):
        tx.run("""
            MERGE (c:Country {iso3: $iso3})
            SET c.name = $name,
                c.regime_type = $regime_type,
                c.region = $region,
                c.subregion = $subregion,
                c.population = $population,
                c.gdp_usd = $gdp_usd,
                c.gdp_per_capita_usd = $gdp_per_capita_usd,
                c.stability_index = $stability_index,
                c.gdp_growth_rate = $gdp_growth_rate,
                c.military_expenditure_usd = $military_expenditure_usd,
                c.nuclear_arsenal = $nuclear_arsenal,
                c.urbanization_rate = $urbanization_rate,
                c.median_age = $median_age,
                c.refugees_hosted = $refugees_hosted,
                c.refugees_origin = $refugees_origin,
                c.idps = $idps,
                c.last_timestep = $timestep,
                c.updated_at = datetime()
        """, 
        iso3=country.iso3,
        name=country.name,
        regime_type=country.regime_type.value if hasattr(country.regime_type, 'value') else country.regime_type,
        region=country.region,
        subregion=country.subregion,
        population=country.population,
        gdp_usd=country.gdp_usd,
        gdp_per_capita_usd=country.gdp_per_capita_usd,
        stability_index=country.stability_index,
        gdp_growth_rate=country.gdp_growth_rate,
        military_expenditure_usd=country.military_expenditure_usd,
        nuclear_arsenal=country.nuclear_arsenal,
        urbanization_rate=country.urbanization_rate,
        median_age=country.median_age,
        refugees_hosted=country.refugees_hosted,
        refugees_origin=country.refugees_origin,
        idps=country.idps,
        timestep=timestep)
    
    @staticmethod
    def _upsert_trade(tx, exporter_iso3, importer_iso3, volume, year):
        tx.run("""
            MERGE (exporter:Country {iso3: $exporter_iso3})
            MERGE (importer:Country {iso3: $importer_iso3})
            MERGE (exporter)-[t:TRADES_WITH {year: $year}]->(importer)
            SET t.value_usd = $volume, t.updated_at = datetime()
        """, exporter_iso3=exporter_iso3, importer_iso3=importer_iso3, volume=volume, year=year)
    
    @staticmethod
    def _upsert_diplomatic(tx, source_iso3, target_iso3, level):
        tx.run("""
            MERGE (source:Country {iso3: $source_iso3})
            MERGE (target:Country {iso3: $target_iso3})
            MERGE (source)-[r:DIPLOMATIC_RELATIONS]->(target)
            SET r.level = $level, r.updated_at = datetime()
        """, source_iso3=source_iso3, target_iso3=target_iso3, level=level)
    
    @staticmethod
    def _upsert_alliance(tx, alliance, timestep):
        tx.run("""
            MERGE (a:Alliance {id: $id})
            SET a.name = $name,
                a.type = $type,
                a.founding_date = $founding_date,
                a.cohesion = $cohesion,
                a.is_active = $is_active,
                a.last_timestep = $timestep,
                a.updated_at = datetime()
            WITH a
            UNWIND $members AS member_iso3
            MERGE (c:Country {iso3: member_iso3})
            MERGE (a)-[:HAS_MEMBER]->(c)
        """,
        id=str(alliance.id),
        name=alliance.name,
        type=alliance.alliance_type.value if hasattr(alliance.alliance_type, 'value') else alliance.alliance_type,
        founding_date=str(alliance.founding_date),
        cohesion=alliance.cohesion,
        is_active=alliance.is_active,
        timestep=timestep,
        members=alliance.members)
    
    @staticmethod
    def _upsert_conflict(tx, conflict, timestep):
        tx.run("""
            MERGE (c:Conflict {id: $id})
            SET c.name = $name,
                c.type = $type,
                c.intensity = $intensity,
                c.battle_deaths = $battle_deaths,
                c.civilian_deaths = $civilian_deaths,
                c.displaced_persons = $displaced_persons,
                c.status = $status,
                c.start_date = $start_date,
                c.end_date = $end_date,
                c.last_timestep = $timestep,
                c.updated_at = datetime()
            WITH c
            MERGE (attacker:Country {iso3: $attacker})
            MERGE (defender:Country {iso3: $defender})
            MERGE (attacker)-[:INVOLVES {role: 'attacker'}]->(c)
            MERGE (defender)-[:INVOLVES {role: 'defender'}]->(c)
        """,
        id=str(conflict.id),
        name=conflict.name,
        type=conflict.conflict_type.value if hasattr(conflict.conflict_type, 'value') else conflict.conflict_type,
        intensity=conflict.intensity.value if hasattr(conflict.intensity, 'value') else conflict.intensity,
        battle_deaths=conflict.battle_deaths,
        civilian_deaths=conflict.civilian_deaths,
        displaced_persons=conflict.displaced_persons,
        status=conflict.status,
        start_date=str(conflict.start_date),
        end_date=str(conflict.end_date) if conflict.end_date else None,
        timestep=timestep,
        attacker=conflict.primary_attacker,
        defender=conflict.primary_defender)
    
    @staticmethod
    def _upsert_treaty(tx, treaty):
        tx.run("""
            MERGE (t:Treaty {id: $id})
            SET t.name = $name,
                t.category = $category,
                t.signed_date = $signed_date,
                t.ratified_date = $ratified_date,
                t.is_active = $is_active,
                t.updated_at = datetime()
            WITH t
            UNWIND $signatories AS signatory_iso3
            MERGE (c:Country {iso3: signatory_iso3})
            MERGE (c)-[:SIGNED {category: $category}]->(t)
            WITH t
            UNWIND $ratifiers AS ratifier_iso3
            MERGE (c2:Country {iso3: ratifier_iso3})
            MERGE (c2)-[:RATIFIED {category: $category}]->(t)
        """,
        id=str(treaty.id),
        name=treaty.name,
        category=treaty.category.value if hasattr(treaty.category, 'value') else treaty.category,
        signed_date=str(treaty.signed_date),
        ratified_date=str(treaty.ratified_date) if treaty.ratified_date else None,
        is_active=treaty.is_active,
        signatories=treaty.signatories,
        ratifiers=treaty.ratifiers)
    
    @staticmethod
    def _upsert_sanction(tx, sanction):
        tx.run("""
            MERGE (s:Sanction {id: $id})
            SET s.name = $name,
                s.type = $type,
                s.status = $status,
                s.imposed_date = $imposed_date,
                s.lifted_date = $lifted_date,
                s.updated_at = datetime()
            WITH s
            MERGE (target:Country {iso3: $target_country})
            MERGE (target)-[:SANCTIONED {type: $type}]->(s)
            WITH s
            UNWIND $imposers AS imposer_iso3
            MERGE (imposer:Country {iso3: imposer_iso3})
            MERGE (imposer)-[:IMPOSED {type: $type}]->(s)
        """,
        id=str(sanction.id),
        name=sanction.name,
        type=sanction.sanction_type.value if hasattr(sanction.sanction_type, 'value') else sanction.sanction_type,
        status=sanction.status,
        imposed_date=str(sanction.imposed_date),
        lifted_date=str(sanction.lifted_date) if sanction.lifted_date else None,
        target_country=sanction.target_country,
        imposers=sanction.imposing_countries)
    
    @staticmethod
    def _upsert_leader(tx, leader):
        tx.run("""
            MERGE (l:Leader {id: $id})
            SET l.name = $name,
                l.title = $title,
                l.ideology = $ideology,
                l.party = $party,
                l.start_date = $start_date,
                l.is_active = $is_active,
                l.updated_at = datetime()
            WITH l
            MERGE (c:Country {iso3: $country_iso3})
            MERGE (c)-[:HAS_LEADER {since: $start_date}]->(l)
        """,
        id=str(leader.id),
        name=leader.name,
        title=leader.title,
        ideology=leader.ideology,
        party=leader.party,
        start_date=str(leader.start_date),
        is_active=leader.is_active,
        country_iso3=leader.country_iso3)
    
    def close(self):
        """Close driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None


def get_exporter(format_name: str, output_dir: str, **kwargs) -> BaseExporter:
    """Factory function to get exporter by format name.
    
    Additional kwargs for specific exporters:
    - kafka: bootstrap_servers, topic_prefix, flush_interval
    - neo4j: uri, user, password, batch_size
    """
    exporters = {
        "parquet": ParquetExporter,
        "csv": CSVExporter,
        "geojson": GeoJSONExporter,
        "network": NetworkExporter,
        "kafka": KafkaExporter,
        "neo4j": Neo4jExporter,
    }
    
    if format_name not in exporters:
        raise ValueError(f"Unknown exporter format: {format_name}. Available: {list(exporters.keys())}")
    
    exporter_class = exporters[format_name]
    return exporter_class(output_dir, **kwargs)