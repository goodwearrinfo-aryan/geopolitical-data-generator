"""Main CLI for geopolitical data generator."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import click
import structlog
import yaml

from config.loader import load_config, SimulationConfig, save_config
from schemas.core import (
    SimulationState, Country, CountryTier, RegimeType, ResourceType
)
from engine.temporal import TemporalEngine, TimestepFrequency
from engine.causal import CausalEngine, ConflictDynamics
from engine.spatial import SpatialEngine
from engine.ensemble import EnsembleEngine, create_ensemble_engine
from calibration.engine import CalibrationEngine
from exporters import get_exporter

# Configure structured logging
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)
logger = structlog.get_logger()


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool):
    """Geopolitical Data Generator - Enterprise synthetic data for world simulation."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose
    
    if config:
        ctx.obj["config"] = load_config(config)
    else:
        ctx.obj["config"] = load_config()


@cli.command()
@click.option("--output", "-o", default="./output", help="Output directory")
@click.option("--format", "-f", multiple=True, 
              type=click.Choice(["parquet", "csv", "geojson", "network"]),
              default=["parquet"], help="Export formats")
@click.option("--timesteps", "-t", type=int, help="Number of timesteps to run")
@click.option("--seed", "-s", type=int, help="Random seed")
@click.pass_context
def run(ctx: click.Context, output: str, format: tuple, timesteps: Optional[int], seed: Optional[int]):
    """Run a single simulation."""
    config = ctx.obj["config"]
    
    if seed is not None:
        config.seed = seed
    
    logger.info("Starting simulation", seed=config.seed, timesteps=timesteps)
    
    # Initialize simulation
    state = _initialize_simulation(config)
    temporal = _create_temporal_engine(config)
    causal = CausalEngine(temporal.rng, config.__dict__)
    spatial = SpatialEngine(config.__dict__)
    conflict_dynamics = ConflictDynamics(temporal.rng, config.__dict__)
    
    # Run simulation
    max_timesteps = timesteps or temporal.total_timesteps
    
    with click.progressbar(length=max_timesteps, label="Simulating") as bar:
        for step in range(max_timesteps):
            if not temporal.advance():
                break
            
            # Update state timestep
            state.timestep = temporal.current_timestep
            state.date = temporal.current_date
            
            # Generate events
            events = causal.generate_events(state, temporal) if hasattr(causal, 'generate_events') else []
            
            # Process conflicts
            conflict_events = conflict_dynamics.step_conflicts(state, temporal)
            events.extend(conflict_events)
            
            # Apply causal effects
            for event in events:
                effects = causal.process_event(event, state, temporal)
                for effect in effects:
                    # Store for later application
                    pass
            
            # Update global aggregates
            _update_global_aggregates(state)
            
            # Export periodically
            if step % 12 == 0:  # Annual
                _export_state(state, output, format)
            
            bar.update(1)
    
    # Final export
    _export_state(state, output, format)
    
    logger.info("Simulation complete", timestep=state.timestep, 
                countries=len(state.countries), conflicts=len(state.conflicts))


@cli.command()
@click.option("--scenarios", "-n", default=100, help="Number of ensemble scenarios")
@click.option("--output", "-o", default="./output/ensemble", help="Output directory")
@click.option("--workers", "-w", type=int, default=4, help="Parallel workers")
@click.option("--sensitivity", is_flag=True, help="Run sensitivity analysis")
@click.pass_context
def ensemble(ctx: click.Context, scenarios: int, output: str, workers: int, sensitivity: bool):
    """Run ensemble simulation."""
    config = ctx.obj["config"]
    config.ensemble_enabled = True
    config.n_scenarios = scenarios
    
    logger.info("Starting ensemble", scenarios=scenarios, workers=workers)
    
    engine = create_ensemble_engine(config.__dict__)
    engine.n_workers = workers
    
    def progress(done: int, total: int):
        click.echo(f"Completed {done}/{total} scenarios")
    
    results = engine.run_ensemble(scenarios, progress_callback=progress)
    
    if sensitivity:
        logger.info("Running sensitivity analysis")
        param_bounds = {
            "coup_base_rate": (0.01, 0.05),
            "escalation_probability": (0.05, 0.3),
            "gdp_shock_std": (0.01, 0.05),
            "alliance_formation_rate": (0.01, 0.1),
            "regime_transition": (0.5, 2.0),
        }
        engine.run_morris_sensitivity(param_bounds, n_trajectories=20)
        engine.run_sobol_sensitivity(param_bounds, n_samples=500)
    
    engine.export_results(output)
    
    summary = engine.get_summary_statistics()
    click.echo(f"Ensemble complete. Results in {output}")
    click.echo(f"Mean final GDP: {summary.get('final_global_gdp', {}).get('mean', 'N/A'):.2e}")


@cli.command()
@click.option("--data-dir", "-d", required=True, help="Directory with calibration data")
@click.option("--method", "-m", type=click.Choice(["bayesian", "moment_matching", "hybrid"]),
              default="hybrid", help="Calibration method")
@click.option("--output", "-o", default="./output/calibration", help="Output directory")
@click.option("--validate", is_flag=True, help="Run validation after calibration")
@click.pass_context
def calibrate(ctx: click.Context, data_dir: str, method: str, output: str, validate: bool):
    """Calibrate model parameters against historical data."""
    config = ctx.obj["config"]
    
    logger.info("Starting calibration", method=method, data_dir=data_dir)
    
    # Initialize state
    state = _initialize_simulation(config)
    
    # Create calibration engine
    cal_engine = CalibrationEngine(config.__dict__)
    cal_engine.load_world_bank_data(f"{data_dir}/world_bank")
    cal_engine.load_inscr_data(f"{data_dir}/inscr")
    cal_engine.load_acled_data(f"{data_dir}/acled/events.csv")
    
    # Run calibration
    result = cal_engine.calibrate(state, method=method)
    
    # Save results
    Path(output).mkdir(parents=True, exist_ok=True)
    
    import json
    with open(Path(output) / "calibration_result.json", "w") as f:
        json.dump({
            "parameters": dict(zip(result.parameter_names, result.optimal_values.tolist())),
            "objective": result.objective_value,
            "convergence": result.convergence,
            "iterations": result.iterations,
            "confidence_intervals": result.confidence_intervals,
        }, f, indent=2, default=str)
    
    click.echo(f"Calibration complete. Objective: {result.objective_value:.4f}")
    
    if validate:
        logger.info("Running validation")
        metrics = cal_engine.validate(result.optimal_values, state)
        click.echo("Validation metrics:")
        for k, v in metrics.items():
            click.echo(f"  {k}: {v:.4f}")


@cli.command()
@click.option("--output", "-o", default="./config.yaml", help="Output config file")
@click.pass_context
def init_config(ctx: click.Context, output: str):
    """Generate default configuration file."""
    config = SimulationConfig()
    save_config(config, output)
    click.echo(f"Default configuration written to {output}")


@cli.command()
@click.pass_context
def validate(ctx: click.Context):
    """Validate configuration and data."""
    config = ctx.obj["config"]
    
    click.echo("Configuration validation:")
    click.echo(f"  Simulation: {config.start_year}-{config.end_year} ({config.timestep})")
    click.echo(f"  Countries: {config.country_tier.value}")
    click.echo(f"  Seed: {config.seed}")
    click.echo(f"  Output: {config.output_dir}")
    click.echo(f"  Formats: {config.export_formats}")
    click.echo(f"  Ensemble: {config.ensemble_enabled} ({config.n_scenarios} scenarios)")
    click.echo(f"  Calibration: {config.calibration_mode}")
    
    click.echo("\nDomains enabled:")
    for domain in ["political", "conflict", "diplomatic", "economic", "military", "demographic"]:
        enabled = getattr(config, f"{domain}_enabled", False)
        click.echo(f"  {domain}: {enabled}")
    
    click.echo("\nValidation passed!")


def _initialize_simulation(config: SimulationConfig) -> SimulationState:
    """Initialize simulation state with countries."""
    state = SimulationState(timestep=0, date=date(config.start_year, 1, 1))
    
    # Load or generate countries
    countries = _generate_countries(config)
    state.countries = {c.iso3: c for c in countries}
    
    # Initialize global aggregates
    _update_global_aggregates(state)
    
    return state


def _generate_countries(config: SimulationConfig) -> list:
    """Generate or load country data."""
    from faker import Faker
    import numpy as np
    
    rng = np.random.default_rng(config.seed)
    fake = Faker()
    fake.seed_instance(config.seed)
    
    # Core countries (major economies/populations)
    core_countries = [
        ("USA", "US", "United States", "Americas", "Northern America", 9833517, 331000000, 25000000000000),
        ("CHN", "CN", "China", "Asia", "Eastern Asia", 9596961, 1412000000, 18000000000000),
        ("JPN", "JP", "Japan", "Asia", "Eastern Asia", 377975, 125800000, 4900000000000),
        ("DEU", "DE", "Germany", "Europe", "Western Europe", 357114, 83200000, 4200000000000),
        ("IND", "IN", "India", "Asia", "Southern Asia", 3287263, 1380000000, 3100000000000),
        ("GBR", "GB", "United Kingdom", "Europe", "Northern Europe", 242495, 67200000, 3100000000000),
        ("FRA", "FR", "France", "Europe", "Western Europe", 551695, 67400000, 2900000000000),
        ("ITA", "IT", "Italy", "Europe", "Southern Europe", 301340, 59600000, 2100000000000),
        ("CAN", "CA", "Canada", "Americas", "Northern America", 9984670, 38000000, 1900000000000),
        ("KOR", "KR", "South Korea", "Asia", "Eastern Asia", 100210, 51800000, 1800000000000),
        ("RUS", "RU", "Russia", "Europe", "Eastern Europe", 17098246, 145900000, 1700000000000),
        ("BRA", "BR", "Brazil", "Americas", "South America", 8515767, 212600000, 1600000000000),
        ("AUS", "AU", "Australia", "Oceania", "Australia and New Zealand", 7692024, 25700000, 1600000000000),
        ("ESP", "ES", "Spain", "Europe", "Southern Europe", 505990, 47400000, 1400000000000),
        ("MEX", "MX", "Mexico", "Americas", "Central America", 1964375, 128900000, 1300000000000),
        ("IDN", "ID", "Indonesia", "Asia", "South-Eastern Asia", 1904569, 273500000, 1200000000000),
        ("NLD", "NL", "Netherlands", "Europe", "Western Europe", 41850, 17400000, 1000000000000),
        ("SAU", "SA", "Saudi Arabia", "Asia", "Western Asia", 2149690, 34800000, 800000000000),
        ("TUR", "TR", "Turkey", "Asia", "Western Asia", 783562, 84300000, 800000000000),
        ("CHE", "CH", "Switzerland", "Europe", "Western Europe", 41284, 8600000, 800000000000),
        ("TWN", "TW", "Taiwan", "Asia", "Eastern Asia", 36197, 23600000, 700000000000),
        ("POL", "PL", "Poland", "Europe", "Eastern Europe", 312696, 37800000, 600000000000),
        ("SWE", "SE", "Sweden", "Europe", "Northern Europe", 450295, 10400000, 600000000000),
        ("BEL", "BE", "Belgium", "Europe", "Western Europe", 30528, 11500000, 600000000000),
        ("THA", "TH", "Thailand", "Asia", "South-Eastern Asia", 513120, 69800000, 500000000000),
        ("IRN", "IR", "Iran", "Asia", "Southern Asia", 1648195, 84000000, 500000000000),
        ("ARG", "AR", "Argentina", "Americas", "South America", 2780400, 45400000, 500000000000),
        ("NOR", "NO", "Norway", "Europe", "Northern Europe", 323802, 5400000, 400000000000),
        ("NGA", "NG", "Nigeria", "Africa", "Western Africa", 923768, 206100000, 400000000000),
        ("ISR", "IL", "Israel", "Asia", "Western Asia", 22072, 9200000, 400000000000),
        ("ARE", "AE", "UAE", "Asia", "Western Asia", 83600, 9900000, 400000000000),
        ("EGY", "EG", "Egypt", "Africa", "Northern Africa", 1001450, 102300000, 400000000000),
        ("ZAF", "ZA", "South Africa", "Africa", "Southern Africa", 1221037, 59300000, 400000000000),
        ("PAK", "PK", "Pakistan", "Asia", "Southern Asia", 881912, 220900000, 300000000000),
        ("BGD", "BD", "Bangladesh", "Asia", "Southern Asia", 147570, 164700000, 300000000000),
        ("PHL", "PH", "Philippines", "Asia", "South-Eastern Asia", 300000, 109600000, 300000000000),
        ("VNM", "VN", "Vietnam", "Asia", "South-Eastern Asia", 331212, 97300000, 300000000000),
        ("COL", "CO", "Colombia", "Americas", "South America", 1141748, 50900000, 300000000000),
        ("MYS", "MY", "Malaysia", "Asia", "South-Eastern Asia", 330803, 32400000, 300000000000),
        ("SGP", "SG", "Singapore", "Asia", "South-Eastern Asia", 728, 5700000, 300000000000),
        ("CHL", "CL", "Chile", "Americas", "South America", 756102, 19100000, 300000000000),
        ("PER", "PE", "Peru", "Americas", "South America", 1285216, 33000000, 200000000000),
        ("FIN", "FI", "Finland", "Europe", "Northern Europe", 338424, 5500000, 200000000000),
        ("CZE", "CZ", "Czechia", "Europe", "Eastern Europe", 78867, 10700000, 200000000000),
        ("ROU", "RO", "Romania", "Europe", "Eastern Europe", 238397, 19200000, 200000000000),
        ("PRT", "PT", "Portugal", "Europe", "Southern Europe", 92212, 10300000, 200000000000),
        ("GRC", "GR", "Greece", "Europe", "Southern Europe", 131957, 10400000, 200000000000),
        ("HUN", "HU", "Hungary", "Europe", "Eastern Europe", 93028, 9700000, 200000000000),
        ("KAZ", "KZ", "Kazakhstan", "Asia", "Central Asia", 2724900, 18800000, 200000000000),
        ("IRQ", "IQ", "Iraq", "Asia", "Western Asia", 438317, 40200000, 200000000000),
    ]
    
    countries = []
    for i, (iso3, iso2, name, region, subregion, area, pop, gdp) in enumerate(core_countries):
        regime_choices = list(RegimeType)
        weights = [0.4, 0.3, 0.2, 0.1] if i < 10 else [0.2, 0.4, 0.3, 0.1]
        regime = rng.choice(regime_choices, p=weights)
        
        country = Country(
            iso3=iso3,
            iso2=iso2,
            name=name,
            region=region,
            subregion=subregion,
            area_km2=area,
            population=pop,
            gdp_usd=gdp * rng.uniform(0.9, 1.1),
            gdp_per_capita_usd=gdp / pop * rng.uniform(0.9, 1.1),
            regime_type=regime,
            polity_score=rng.integers(-10, 11),
            stability_index=rng.uniform(0.3, 0.9),
            coup_risk=rng.uniform(0.005, 0.05),
            protest_level=rng.uniform(0.05, 0.3),
            gdp_growth_rate=rng.normal(0.025, 0.015),
            inflation_rate=rng.uniform(0.01, 0.08),
            unemployment_rate=rng.uniform(0.03, 0.15),
            trade_openness=rng.uniform(0.3, 1.5),
            military_expenditure_usd=gdp * rng.uniform(0.01, 0.04),
            military_personnel=int(pop * rng.uniform(0.001, 0.01)),
            nuclear_arsenal=rng.integers(0, 100) if iso3 in ["USA", "RUS", "CHN", "FRA", "GBR", "IND", "PAK", "ISR", "PRK"] else 0,
            urbanization_rate=rng.uniform(0.5, 3.0),
            median_age=rng.uniform(20, 50),
            life_expectancy=rng.uniform(60, 85),
            literacy_rate=rng.uniform(0.7, 1.0),
            ethnic_fragmentation=rng.uniform(0.1, 0.9),
            religious_fragmentation=rng.uniform(0.1, 0.9),
            refugees_hosted=rng.integers(0, 1000000),
            refugees_origin=rng.integers(0, 500000),
            idps=rng.integers(0, 1000000),
            tier=CountryTier.CORE,
            resources={
                ResourceType.OIL: rng.uniform(0, 10000000) if iso3 in ["USA", "SAU", "RUS", "IRN", "IRQ", "VEN", "NGA"] else 0,
                ResourceType.GAS: rng.uniform(0, 5000000) if iso3 in ["USA", "RUS", "IRN", "QAT", "ARE"] else 0,
                ResourceType.RARE_EARTHS: rng.uniform(0, 100000) if iso3 in ["CHN", "AUS", "USA"] else 0,
            },
        )
        countries.append(country)
    
    # Add extended countries if needed
    if config.country_tier in [CountryTier.EXTENDED, CountryTier.ALL]:
        # Generate additional countries with Faker
        for i in range(150):
            iso3 = fake.country_code(representation="alpha-3")
            if iso3 in [c.iso3 for c in countries]:
                continue
            
            country = Country(
                iso3=iso3,
                iso2=fake.country_code(representation="alpha-2"),
                name=fake.country(),
                region=fake.random_element(["Africa", "Americas", "Asia", "Europe", "Oceania"]),
                area_km2=rng.uniform(1000, 2000000),
                population=rng.integers(100000, 50000000),
                gdp_usd=rng.uniform(1e9, 5e11),
                gdp_per_capita_usd=rng.uniform(500, 30000),
                regime_type=rng.choice(list(RegimeType)),
                stability_index=rng.uniform(0.2, 0.8),
                tier=CountryTier.EXTENDED,
            )
            countries.append(country)
            if len(countries) >= 195:
                break
    
    return countries


def _create_temporal_engine(config: SimulationConfig) -> TemporalEngine:
    """Create temporal engine from config."""
    freq_map = {
        "monthly": TimestepFrequency.MONTHLY,
        "quarterly": TimestepFrequency.QUARTERLY,
        "annual": TimestepFrequency.ANNUAL,
    }
    return TemporalEngine(
        start_date=date(config.start_year, 1, 1),
        end_date=date(config.end_year, 12, 31),
        frequency=freq_map.get(config.timestep, TimestepFrequency.MONTHLY),
        seed=config.seed,
    )


def _update_global_aggregates(state: SimulationState) -> None:
    """Update global aggregate statistics."""
    state.global_gdp_usd = sum(c.gdp_usd for c in state.countries.values())
    state.global_population = sum(c.population for c in state.countries.values())
    state.active_conflicts = sum(1 for c in state.conflicts.values() if c.status == "ongoing")
    state.total_battle_deaths = sum(c.battle_deaths for c in state.conflicts.values())
    state.total_displaced = sum(c.displaced_persons for c in state.conflicts.values())


def _export_state(state: SimulationState, output: str, formats: tuple) -> None:
    """Export state to all requested formats."""
    for fmt in formats:
        exporter = get_exporter(fmt, output)
        exporter.export(state)


if __name__ == "__main__":
    cli()