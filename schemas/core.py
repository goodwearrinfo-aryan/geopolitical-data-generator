"""Core data schemas for geopolitical simulation."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class CountryTier(str, Enum):
    CORE = "core"
    EXTENDED = "extended"
    ALL = "all"


class RegimeType(str, Enum):
    DEMOCRACY = "democracy"
    AUTOCRACY = "autocracy"
    ANOCRACY = "anocracy"
    FAILED_STATE = "failed_state"


class ConflictType(str, Enum):
    INTERSTATE = "interstate"
    CIVIL = "civil"
    ETHNIC = "ethnic"
    TERRORISM = "terrorism"
    PROXY = "proxy"


class ConflictIntensity(str, Enum):
    TENSION = "tension"
    SKIRMISH = "skirmish"
    LOCALIZED = "localized"
    MAJOR = "major"
    TOTAL = "total"


class AllianceType(str, Enum):
    DEFENSE = "defense"
    ECONOMIC = "economic"
    CULTURAL = "cultural"
    INTELLIGENCE = "intelligence"


class SanctionType(str, Enum):
    TRADE = "trade"
    FINANCIAL = "financial"
    TRAVEL = "travel"
    ARMS = "arms"
    COMPREHENSIVE = "comprehensive"


class TreatyCategory(str, Enum):
    TRADE = "trade"
    ENVIRONMENT = "environment"
    SECURITY = "security"
    HUMAN_RIGHTS = "human_rights"
    NUCLEAR = "nuclear"


class ResourceType(str, Enum):
    OIL = "oil"
    GAS = "gas"
    COAL = "coal"
    RARE_EARTHS = "rare_earths"
    WATER = "water"
    ARABLE_LAND = "arable_land"


class EventType(str, Enum):
    ELECTION = "election"
    COUP = "coup"
    PROTEST = "protest"
    REGIME_CHANGE = "regime_change"
    LEADER_CHANGE = "leader_change"
    POLICY_SHIFT = "policy_shift"
    CONFLICT_START = "conflict_start"
    CONFLICT_ESCALATION = "conflict_escalation"
    CONFLICT_DEESCALATION = "conflict_deescalation"
    CONFLICT_END = "conflict_end"
    TREATY_SIGNED = "treaty_signed"
    TREATY_RATIFIED = "treaty_ratified"
    TREATY_WITHDRAWAL = "treaty_withdrawal"
    SANCTION_IMPOSED = "sanction_imposed"
    SANCTION_LIFTED = "sanction_lifted"
    ALLIANCE_FORMED = "alliance_formed"
    ALLIANCE_DISSOLVED = "alliance_dissolved"
    WAR_DECLARATION = "war_declaration"
    CEASEFIRE = "ceasefire"
    PEACE_TREATY = "peace_treaty"
    REFUGEE_CRISIS = "refugee_crisis"
    NATURAL_DISASTER = "natural_disaster"
    ECONOMIC_CRISIS = "economic_crisis"
    PANDEMIC = "pandemic"
    CYBER_ATTACK = "cyber_attack"
    NUCLEAR_TEST = "nuclear_test"
    MILITARY_EXERCISE = "military_exercise"
    ARMS_DEAL = "arms_deal"
    LEADER_MEETING = "leader_meeting"
    SUMMIT = "summit"
    EMBASSY_OPEN = "embassy_open"
    EMBASSY_CLOSE = "embassy_close"


class BaseEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Country(BaseEntity):
    iso3: str = Field(pattern=r"^[A-Z]{3}$")
    iso2: str = Field(pattern=r"^[A-Z]{2}$")
    name: str
    official_name: Optional[str] = None
    capital: Optional[str] = None
    region: str  # UN geoscheme
    subregion: Optional[str] = None
    
    # Geography
    area_km2: float
    population: int
    gdp_usd: float
    gdp_per_capita_usd: float
    
    # Political
    regime_type: RegimeType
    polity_score: Optional[int] = Field(default=None, ge=-10, le=10)  # -10 to 10
    leader_id: Optional[UUID] = None
    ruling_party: Optional[str] = None
    election_cycle_years: int = 4
    next_election_year: Optional[int] = None
    
    # Stability
    stability_index: float = Field(default=0.5, ge=0.0, le=1.0)
    state_fragility_index: Optional[float] = Field(default=None, ge=0.0, le=25.0)
    coup_risk: float = Field(default=0.02, ge=0.0, le=1.0)
    protest_level: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Economy
    gdp_growth_rate: float = 0.0
    inflation_rate: float = 0.0
    unemployment_rate: float = 0.0
    debt_to_gdp: float = 0.0
    trade_openness: float = 0.0
    currency: str = "USD"
    fx_reserves_usd: float = 0.0
    
    # Resources
    resources: Dict[ResourceType, float] = Field(default_factory=dict)
    resource_exports: Dict[ResourceType, float] = Field(default_factory=dict)
    
    # Military
    military_expenditure_usd: float = 0.0
    military_personnel: int = 0
    nuclear_arsenal: int = 0
    military_alliances: List[str] = Field(default_factory=list)
    
    # Demographics
    urbanization_rate: float = 0.0
    median_age: float = 0.0
    life_expectancy: float = 0.0
    literacy_rate: float = 0.0
    ethnic_fragmentation: float = 0.0
    religious_fragmentation: float = 0.0
    refugees_hosted: int = 0
    refugees_origin: int = 0
    idps: int = 0  # internally displaced persons
    
    # Relations
    diplomatic_relations: Dict[str, int] = Field(default_factory=dict)  # iso3 -> level (-100 to 100)
    trade_partners: Dict[str, float] = Field(default_factory=dict)  # iso3 -> trade volume USD
    sanctions_on: List[str] = Field(default_factory=list)  # sanction IDs
    sanctions_by: List[str] = Field(default_factory=list)  # sanction IDs
    
    # Metadata
    tier: CountryTier = CountryTier.CORE
    data_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    last_calibrated: Optional[datetime] = None


class Leader(BaseEntity):
    country_iso3: str
    name: str
    title: str  # President, Prime Minister, Monarch, etc.
    birth_year: int
    gender: Literal["M", "F", "O"]
    party: Optional[str] = None
    ideology: Optional[str] = None  # left, right, center, nationalist, etc.
    
    # Dates
    start_date: date
    end_date: Optional[date] = None
    term_limit_years: Optional[int] = None
    
    # Attributes
    competence: float = Field(default=0.5, ge=0.0, le=1.0)
    charisma: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    hawkishness: float = Field(default=0.5, ge=0.0, le=1.0)
    corruption: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Background
    military_background: bool = False
    education_level: Literal["primary", "secondary", "tertiary", "postgraduate"] = "tertiary"
    previous_roles: List[str] = Field(default_factory=list)
    
    # Status
    is_active: bool = True
    exit_reason: Optional[str] = None  # election, coup, death, resignation, term_limit


class PoliticalEvent(BaseEntity):
    country_iso3: str
    event_type: EventType
    date: date
    description: str
    
    # Participants
    actors: List[str] = Field(default_factory=list)  # party names, group names, leader IDs
    
    # Impact
    regime_change: bool = False
    new_regime_type: Optional[RegimeType] = None
    stability_impact: float = 0.0  # -1 to 1
    legitimacy_impact: float = 0.0
    
    # Details
    casualties: int = 0
    participants_estimate: int = 0
    duration_days: int = 1
    outcome: Optional[str] = None  # success, failure, ongoing, negotiated
    
    # Metadata
    source: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)


class Conflict(BaseEntity):
    name: str
    conflict_type: ConflictType
    intensity: ConflictIntensity
    
    # Participants
    primary_attacker: str  # ISO3
    primary_defender: str  # ISO3
    secondary_participants: Dict[str, Literal["attacker", "defender", "mediator", "peacekeeper"]] = Field(default_factory=dict)
    
    # Timeline
    start_date: date
    end_date: Optional[date] = None
    status: Literal["ongoing", "ceasefire", "frozen", "ended", "escalated"] = "ongoing"
    
    # Geography
    theater_countries: List[str] = Field(default_factory=list)  # ISO3 codes
    contested_territories: List[str] = Field(default_factory=list)
    
    # Casualties & Impact
    battle_deaths: int = 0
    civilian_deaths: int = 0
    displaced_persons: int = 0
    infrastructure_damage_usd: float = 0.0
    economic_cost_usd: float = 0.0
    
    # Escalation
    escalation_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    nuclear_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    chemical_weapons_used: bool = False
    
    # Resolution
    peace_treaty_id: Optional[UUID] = None
    mediator_countries: List[str] = Field(default_factory=list)
    un_mission: bool = False


class Alliance(BaseEntity):
    name: str
    alliance_type: AllianceType
    members: List[str] = Field(default_factory=list)  # ISO3 codes
    founding_date: date
    treaty_id: Optional[UUID] = None
    
    # Obligations
    mutual_defense: bool = False
    consultation_required: bool = False
    joint_exercises: bool = False
    intelligence_sharing: bool = False
    economic_integration: bool = False
    
    # Status
    is_active: bool = True
    cohesion: float = Field(default=0.7, ge=0.0, le=1.0)
    dissolved_date: Optional[date] = None
    dissolution_reason: Optional[str] = None


class Treaty(BaseEntity):
    name: str
    category: TreatyCategory
    signatories: List[str] = Field(default_factory=list)  # ISO3 codes
    ratifiers: List[str] = Field(default_factory=list)  # ISO3 codes
    
    # Dates
    signed_date: date
    ratified_date: Optional[date] = None
    entered_force_date: Optional[date] = None
    expiry_date: Optional[date] = None
    
    # Content
    articles: List[str] = Field(default_factory=list)
    verification_mechanism: Optional[str] = None
    dispute_resolution: Optional[str] = None
    
    # Status
    is_active: bool = True
    withdrawals: Dict[str, date] = Field(default_factory=dict)  # ISO3 -> date
    violations: List[str] = Field(default_factory=list)  # violation descriptions
    
    # Related
    alliance_id: Optional[UUID] = None
    sanction_id: Optional[UUID] = None


class Sanction(BaseEntity):
    name: str
    sanction_type: SanctionType
    target_country: str  # ISO3
    imposing_countries: List[str] = Field(default_factory=list)  # ISO3 codes
    imposing_organizations: List[str] = Field(default_factory=list)  # UN, EU, etc.
    
    # Timeline
    imposed_date: date
    lifted_date: Optional[date] = None
    review_date: Optional[date] = None
    status: Literal["active", "lifted", "suspended", "expired"] = "active"
    
    # Scope
    sectors: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)  # specific companies, individuals
    individuals: List[str] = Field(default_factory=list)  # specific persons
    financial_restrictions: bool = False
    trade_restrictions: bool = False
    travel_bans: bool = False
    arms_embargo: bool = False
    
    # Impact
    estimated_cost_target_usd: float = 0.0
    estimated_cost_imposers_usd: float = 0.0
    compliance_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Legal
    legal_basis: Optional[str] = None
    un_resolution: Optional[str] = None


class EconomicIndicator(BaseEntity):
    country_iso3: str
    year: int
    quarter: Optional[int] = None
    month: Optional[int] = None
    
    # National Accounts
    gdp_nominal_usd: Optional[float] = None
    gdp_ppp_usd: Optional[float] = None
    gdp_growth_pct: Optional[float] = None
    gdp_per_capita_usd: Optional[float] = None
    
    # Components
    consumption_pct_gdp: Optional[float] = None
    investment_pct_gdp: Optional[float] = None
    government_pct_gdp: Optional[float] = None
    exports_pct_gdp: Optional[float] = None
    imports_pct_gdp: Optional[float] = None
    
    # Prices & Money
    inflation_pct: Optional[float] = None
    cpi_index: Optional[float] = None
    ppi_index: Optional[float] = None
    exchange_rate_usd: Optional[float] = None
    interest_rate_pct: Optional[float] = None
    money_supply_growth_pct: Optional[float] = None
    
    # Labor
    unemployment_pct: Optional[float] = None
    labor_force_participation_pct: Optional[float] = None
    youth_unemployment_pct: Optional[float] = None
    
    # Fiscal
    revenue_pct_gdp: Optional[float] = None
    expenditure_pct_gdp: Optional[float] = None
    deficit_pct_gdp: Optional[float] = None
    debt_pct_gdp: Optional[float] = None
    
    # Trade
    trade_balance_usd: Optional[float] = None
    current_account_usd: Optional[float] = None
    fdi_inflow_usd: Optional[float] = None
    fdi_outflow_usd: Optional[float] = None
    reserves_usd: Optional[float] = None
    
    # Energy & Resources
    energy_consumption_per_capita: Optional[float] = None
    co2_emissions_per_capita: Optional[float] = None
    renewable_pct_energy: Optional[float] = None
    
    # Inequality & Development
    gini_coefficient: Optional[float] = None
    hdi: Optional[float] = None
    poverty_rate: Optional[float] = None
    
    # Metadata
    source: str = "synthetic"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TradeFlow(BaseEntity):
    year: int
    exporter_iso3: str
    importer_iso3: str
    product_code: str  # HS code
    product_category: str
    value_usd: float
    quantity: Optional[float] = None
    quantity_unit: Optional[str] = None


class MigrationFlow(BaseEntity):
    year: int
    origin_iso3: str
    destination_iso3: str
    migrants: int
    flow_type: Literal["refugee", "asylum_seeker", "economic", "family", "student", "irregular"]
    demographic_breakdown: Optional[Dict[str, int]] = None  # age/sex groups


class DemographicProfile(BaseEntity):
    country_iso3: str
    year: int
    
    # Population
    total_population: int
    male_population: int
    female_population: int
    population_growth_rate: float
    population_density: float  # per km2
    
    # Age Structure
    age_0_14_pct: float
    age_15_64_pct: float
    age_65_plus_pct: float
    median_age: float
    dependency_ratio: float
    
    # Urbanization
    urban_population_pct: float
    urbanization_rate: float
    major_cities: List[Dict[str, Any]] = Field(default_factory=list)  # name, population
    
    # Health
    life_expectancy_total: float
    life_expectancy_male: float
    life_expectancy_female: float
    infant_mortality_per_1000: float
    maternal_mortality_per_100k: float
    fertility_rate: float
    contraceptive_prevalence: float
    hiv_prevalence: float
    
    # Education
    literacy_rate_total: float
    literacy_rate_male: float
    literacy_rate_female: float
    primary_enrollment_pct: float
    secondary_enrollment_pct: float
    tertiary_enrollment_pct: float
    mean_years_schooling: float
    
    # Migration
    net_migration_rate: float
    refugee_stock: int
    asylum_seekers_pending: int
    idp_stock: int
    remittances_received_usd: float
    
    # Composition
    ethnic_groups: List[Dict[str, Any]] = Field(default_factory=list)  # name, pct
    religious_groups: List[Dict[str, Any]] = Field(default_factory=list)  # name, pct
    languages: List[Dict[str, Any]] = Field(default_factory=list)  # name, pct, official
    
    # Labor
    labor_force_total: int
    labor_force_female_pct: float
    employment_by_sector: Dict[str, float] = Field(default_factory=dict)  # agriculture, industry, services
    vulnerable_employment_pct: float
    
    # Metadata
    source: str = "synthetic"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SimulationState(BaseModel):
    """Complete simulation state at a given timestep."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestep: int
    date: date
    countries: Dict[str, Country] = Field(default_factory=dict)
    leaders: Dict[UUID, Leader] = Field(default_factory=dict)
    conflicts: Dict[UUID, Conflict] = Field(default_factory=dict)
    alliances: Dict[UUID, Alliance] = Field(default_factory=dict)
    treaties: Dict[UUID, Treaty] = Field(default_factory=dict)
    sanctions: Dict[UUID, Sanction] = Field(default_factory=dict)
    events: List[PoliticalEvent] = Field(default_factory=list)
    economic_indicators: List[EconomicIndicator] = Field(default_factory=list)
    trade_flows: List[TradeFlow] = Field(default_factory=list)
    migration_flows: List[MigrationFlow] = Field(default_factory=list)
    demographic_profiles: List[DemographicProfile] = Field(default_factory=list)
    
    # Global aggregates
    global_gdp_usd: float = 0.0
    global_population: int = 0
    global_co2_emissions: float = 0.0
    active_conflicts: int = 0
    total_battle_deaths: int = 0
    total_displaced: int = 0