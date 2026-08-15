#!/usr/bin/env python3
"""Generate synthetic World Bank WDI calibration fixtures.

Outputs CSV files matching World Bank WDI schema for:
- GDP (nominal, PPP, per capita, growth)
- Population (total, urban %)
- Trade (exports, imports, partners)
- Economic indicators (inflation, unemployment, debt)
- Energy/Environment (CO2, energy use, renewables)

Usage:
    python tests/generate_wdi.py --countries 50 --years 10 --seed 42
    python tests/generate_wdi.py --countries 195 --years 30 --seed 42 --output tests/fixtures/world_bank/v2
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
from faker import Faker

# Core countries with realistic base values (ISO3, name, region, population, gdp_usd)
CORE_COUNTRIES = [
    ("USA", "United States", "Americas", "Northern America", 331000000, 25000000000000),
    ("CHN", "China", "Asia", "Eastern Asia", 1412000000, 18000000000000),
    ("JPN", "Japan", "Asia", "Eastern Asia", 125800000, 4900000000000),
    ("DEU", "Germany", "Europe", "Western Europe", 83200000, 4200000000000),
    ("IND", "India", "Asia", "Southern Asia", 1380000000, 3100000000000),
    ("GBR", "United Kingdom", "Europe", "Northern Europe", 67200000, 3100000000000),
    ("FRA", "France", "Europe", "Western Europe", 67400000, 2900000000000),
    ("ITA", "Italy", "Europe", "Southern Europe", 59600000, 2100000000000),
    ("CAN", "Canada", "Americas", "Northern America", 38000000, 1900000000000),
    ("KOR", "South Korea", "Asia", "Eastern Asia", 51800000, 1800000000000),
    ("RUS", "Russia", "Europe", "Eastern Europe", 145900000, 1700000000000),
    ("BRA", "Brazil", "Americas", "South America", 212600000, 1600000000000),
    ("AUS", "Australia", "Oceania", "Australia and New Zealand", 25700000, 1600000000000),
    ("ESP", "Spain", "Europe", "Southern Europe", 47400000, 1400000000000),
    ("MEX", "Mexico", "Americas", "Central America", 128900000, 1300000000000),
    ("IDN", "Indonesia", "Asia", "South-Eastern Asia", 273500000, 1200000000000),
    ("NLD", "Netherlands", "Europe", "Western Europe", 17400000, 1000000000000),
    ("SAU", "Saudi Arabia", "Asia", "Western Asia", 34800000, 800000000000),
    ("TUR", "Turkey", "Asia", "Western Asia", 84300000, 800000000000),
    ("CHE", "Switzerland", "Europe", "Western Europe", 8600000, 800000000000),
    ("TWN", "Taiwan", "Asia", "Eastern Asia", 23600000, 700000000000),
    ("POL", "Poland", "Europe", "Eastern Europe", 37800000, 600000000000),
    ("SWE", "Sweden", "Europe", "Northern Europe", 10400000, 600000000000),
    ("BEL", "Belgium", "Europe", "Western Europe", 11500000, 600000000000),
    ("THA", "Thailand", "Asia", "South-Eastern Asia", 69800000, 500000000000),
    ("IRN", "Iran", "Asia", "Southern Asia", 84000000, 500000000000),
    ("ARG", "Argentina", "Americas", "South America", 45400000, 500000000000),
    ("NOR", "Norway", "Europe", "Northern Europe", 5400000, 400000000000),
    ("NGA", "Nigeria", "Africa", "Western Africa", 206100000, 400000000000),
    ("ISR", "Israel", "Asia", "Western Asia", 9200000, 400000000000),
    ("ARE", "UAE", "Asia", "Western Asia", 9900000, 400000000000),
    ("EGY", "Egypt", "Africa", "Northern Africa", 102300000, 400000000000),
    ("ZAF", "South Africa", "Africa", "Southern Africa", 59300000, 400000000000),
    ("PAK", "Pakistan", "Asia", "Southern Asia", 220900000, 300000000000),
    ("BGD", "Bangladesh", "Asia", "Southern Asia", 164700000, 300000000000),
    ("PHL", "Philippines", "Asia", "South-Eastern Asia", 109600000, 300000000000),
    ("VNM", "Vietnam", "Asia", "South-Eastern Asia", 97300000, 300000000000),
    ("COL", "Colombia", "Americas", "South America", 50900000, 300000000000),
    ("MYS", "Malaysia", "Asia", "South-Eastern Asia", 32400000, 300000000000),
    ("SGP", "Singapore", "Asia", "South-Eastern Asia", 5700000, 300000000000),
    ("CHL", "Chile", "Americas", "South America", 19100000, 300000000000),
    ("PER", "Peru", "Americas", "South America", 33000000, 200000000000),
    ("FIN", "Finland", "Europe", "Northern Europe", 5500000, 200000000000),
    ("CZE", "Czechia", "Europe", "Eastern Europe", 10700000, 200000000000),
    ("ROU", "Romania", "Europe", "Eastern Europe", 19200000, 200000000000),
    ("PRT", "Portugal", "Europe", "Southern Europe", 10300000, 200000000000),
    ("GRC", "Greece", "Europe", "Southern Europe", 10400000, 200000000000),
    ("HUN", "Hungary", "Europe", "Eastern Europe", 9700000, 200000000000),
    ("KAZ", "Kazakhstan", "Asia", "Central Asia", 18800000, 200000000000),
    ("IRQ", "Iraq", "Asia", "Western Asia", 40200000, 200000000000),
]

EXTENDED_REGIONS = [
    ("Africa", ["Eastern Africa", "Middle Africa", "Northern Africa", "Southern Africa", "Western Africa"]),
    ("Americas", ["Caribbean", "Central America", "Northern America", "South America"]),
    ("Asia", ["Central Asia", "Eastern Asia", "South-Eastern Asia", "Southern Asia", "Western Asia"]),
    ("Europe", ["Eastern Europe", "Northern Europe", "Southern Europe", "Western Europe"]),
    ("Oceania", ["Australia and New Zealand", "Melanesia", "Micronesia", "Polynesia"]),
]


def generate_countries(n_countries: int, seed: int, core_only: bool = True) -> List[Dict]:
    """Generate country metadata with realistic base values."""
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    countries = []

    # Start with core countries
    for iso3, name, region, subregion, pop, gdp in CORE_COUNTRIES:
        if len(countries) >= n_countries:
            break
        countries.append({
            "iso3": iso3,
            "name": name,
            "region": region,
            "subregion": subregion,
            "base_population": pop,
            "base_gdp_usd": gdp,
            "is_core": True,
        })

    # Add extended countries if needed
    if not core_only and len(countries) < n_countries:
        remaining = n_countries - len(countries)
        for i in range(remaining):
            region, subregions = rng.choice(EXTENDED_REGIONS)
            subregion = rng.choice(subregions)
            iso3 = fake.unique.country_code(representation="alpha-3")
            name = fake.unique.country()
            
            # Random but realistic population and GDP
            pop = rng.lognormal(15, 1.5)  # ~3M to 100M
            pop = int(np.clip(pop, 100000, 200000000))
            gdp_per_capita = rng.lognormal(8, 1.2)  # ~$3K to $30K
            gdp = pop * gdp_per_capita
            
            countries.append({
                "iso3": iso3,
                "name": name,
                "region": region,
                "subregion": subregion,
                "base_population": pop,
                "base_gdp_usd": gdp,
                "is_core": False,
            })

    return countries[:n_countries]


def generate_gdp_data(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate GDP time series with realistic growth patterns."""
    rows = []
    for c in countries:
        gdp = c["base_gdp_usd"]
        pop = c["base_population"]
        
        # Country-specific growth trend
        if c["is_core"]:
            trend = rng.normal(0.025, 0.008)  # 1.5-3.5% for developed
        else:
            trend = rng.normal(0.045, 0.015)  # 2-6% for developing
        
        # Volatility
        vol = rng.uniform(0.01, 0.03)
        
        for year in years:
            # Growth with shock
            shock = rng.normal(0, vol)
            growth = trend + shock
            gdp *= (1 + growth)
            pop *= (1 + rng.normal(0.01, 0.005))
            
            gdp_per_capita = gdp / pop
            ppp_factor = rng.uniform(0.6, 1.4)  # PPP adjustment
            
            rows.append({
                "country": c["name"],
                "iso3": c["iso3"],
                "year": year,
                "gdp_usd": round(gdp, 2),
                "gdp_ppp_usd": round(gdp * ppp_factor, 2),
                "gdp_per_capita_usd": round(gdp_per_capita, 2),
                "gdp_per_capita_ppp_usd": round(gdp_per_capita * ppp_factor, 2),
                "gdp_growth_pct": round(growth * 100, 2),
            })
    return rows


def generate_population_data(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate population and urbanization data."""
    rows = []
    for c in countries:
        pop = c["base_population"]
        urban_pct = rng.uniform(30, 95) if c["is_core"] else rng.uniform(20, 80)
        urban_trend = rng.uniform(0.2, 0.8)  # % per year
        
        for year in years:
            pop *= (1 + rng.normal(0.01, 0.005))
            urban_pct = min(95, urban_pct + urban_trend + rng.normal(0, 0.1))
            
            rows.append({
                "country": c["name"],
                "iso3": c["iso3"],
                "year": year,
                "population": int(pop),
                "urban_population_pct": round(urban_pct, 2),
                "population_growth_pct": round(rng.normal(1.0, 0.5), 2),
                "population_density": round(pop / rng.uniform(1000, 10000000), 2),
            })
    return rows


def generate_trade_data(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate bilateral trade flows using gravity model."""
    rows = []
    iso3_list = [c["iso3"] for c in countries]
    
    # Pre-compute distances (simplified)
    distances = {}
    for i, a in enumerate(iso3_list):
        for b in iso3_list[i+1:]:
            distances[(a, b)] = distances[(b, a)] = rng.uniform(100, 20000)
    
    for year in years:
        for c in countries:
            # Total trade as % of GDP
            trade_openness = rng.uniform(0.3, 1.5) if c["is_core"] else rng.uniform(0.5, 2.0)
            total_trade = c["base_gdp_usd"] * trade_openness
            
            # Distribute among partners (top 10)
            n_partners = min(10, len(countries) - 1)
            partners = rng.choice([x for x in iso3_list if x != c["iso3"]], n_partners, replace=False)
            
            for partner_iso3 in partners:
                # Gravity: GDP product / distance^1.5
                partner = next(p for p in countries if p["iso3"] == partner_iso3)
                dist = distances.get((c["iso3"], partner_iso3), rng.uniform(1000, 15000))
                gravity = (c["base_gdp_usd"] * partner["base_gdp_usd"]) / (dist ** 1.5)
                
                # Normalize to total trade
                export_share = gravity / sum(
                    (c["base_gdp_usd"] * countries[j]["base_gdp_usd"]) / 
                    (distances.get((c["iso3"], countries[j]["iso3"]), 5000) ** 1.5)
                    for j in range(len(countries)) if countries[j]["iso3"] != c["iso3"]
                )
                
                export_value = total_trade * export_share * rng.uniform(0.8, 1.2)
                import_value = export_value * rng.uniform(0.7, 1.3)
                
                # HS product code (2-digit)
                product_code = f"{rng.integers(1, 99):02d}"
                
                rows.append({
                    "year": year,
                    "exporter_iso3": c["iso3"],
                    "importer_iso3": partner_iso3,
                    "product_code": product_code,
                    "export_value_usd": round(export_value, 2),
                    "import_value_usd": round(import_value, 2),
                })
    return rows


def generate_economic_indicators(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate macroeconomic indicators."""
    rows = []
    for c in countries:
        for year in years:
            # Inflation
            if c["is_core"]:
                inflation = rng.normal(0.02, 0.01)
            else:
                inflation = rng.normal(0.05, 0.03)
            inflation = max(-0.02, inflation)
            
            # Unemployment
            unemployment = rng.uniform(3, 12) if c["is_core"] else rng.uniform(5, 25)
            
            # Debt to GDP
            debt_gdp = rng.uniform(30, 130) if c["is_core"] else rng.uniform(20, 80)
            
            # Fiscal
            revenue_gdp = rng.uniform(15, 45)
            expenditure_gdp = revenue_gdp + rng.uniform(-5, 5)
            
            # Current account
            ca_gdp = rng.uniform(-5, 10)
            
            # FDI
            fdi_inflow = c["base_gdp_usd"] * rng.uniform(0.005, 0.05)
            fdi_outflow = c["base_gdp_usd"] * rng.uniform(0.001, 0.03)
            
            rows.append({
                "country": c["name"],
                "iso3": c["iso3"],
                "year": year,
                "inflation_pct": round(inflation * 100, 2),
                "unemployment_pct": round(unemployment, 2),
                "debt_to_gdp_pct": round(debt_gdp, 2),
                "revenue_pct_gdp": round(revenue_gdp, 2),
                "expenditure_pct_gdp": round(expenditure_gdp, 2),
                "current_account_pct_gdp": round(ca_gdp, 2),
                "fdi_inflow_usd": round(fdi_inflow, 2),
                "fdi_outflow_usd": round(fdi_outflow, 2),
                "reserves_usd": round(c["base_gdp_usd"] * rng.uniform(0.05, 0.3), 2),
            })
    return rows


def generate_energy_data(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate energy and emissions data."""
    rows = []
    for c in countries:
        for year in years:
            energy_per_capita = rng.uniform(1, 8) if c["is_core"] else rng.uniform(0.3, 3)
            co2_per_capita = energy_per_capita * rng.uniform(0.2, 0.5)
            renewable_pct = rng.uniform(5, 50) if c["is_core"] else rng.uniform(5, 30)
            
            rows.append({
                "country": c["name"],
                "iso3": c["iso3"],
                "year": year,
                "energy_use_per_capita": round(energy_per_capita, 3),
                "co2_emissions_per_capita": round(co2_per_capita, 3),
                "renewable_energy_pct": round(renewable_pct, 2),
                "electricity_access_pct": round(rng.uniform(80, 100) if c["is_core"] else rng.uniform(40, 100), 2),
            })
    return rows


def write_csv(rows: List[Dict], filepath: Path, fieldnames: List[str]):
    """Write rows to CSV with consistent formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic WDI calibration fixtures")
    parser.add_argument("--countries", type=int, default=50, help="Number of countries")
    parser.add_argument("--years", type=int, default=10, help="Number of years of history")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year")
    parser.add_argument("--core-only", action="store_true", default=True, help="Only core countries")
    parser.add_argument("--output", type=str, default="tests/fixtures/world_bank/v1", help="Output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    years = list(range(args.start_year, args.start_year + args.years))
    countries = generate_countries(args.countries, args.seed, args.core_only)
    output_dir = Path(args.output)

    print(f"Generating WDI fixtures for {len(countries)} countries, {len(years)} years ({years[0]}-{years[-1]})")

    # Generate all datasets
    gdp_data = generate_gdp_data(countries, years, rng)
    pop_data = generate_population_data(countries, years, rng)
    trade_data = generate_trade_data(countries, years, rng)
    econ_data = generate_economic_indicators(countries, years, rng)
    energy_data = generate_energy_data(countries, years, rng)

    # Write CSVs
    write_csv(gdp_data, output_dir / "gdp.csv", [
        "country", "iso3", "year", "gdp_usd", "gdp_ppp_usd",
        "gdp_per_capita_usd", "gdp_per_capita_ppp_usd", "gdp_growth_pct"
    ])
    
    write_csv(pop_data, output_dir / "population.csv", [
        "country", "iso3", "year", "population", "urban_population_pct",
        "population_growth_pct", "population_density"
    ])
    
    write_csv(trade_data, output_dir / "trade.csv", [
        "year", "exporter_iso3", "importer_iso3", "product_code",
        "export_value_usd", "import_value_usd"
    ])
    
    write_csv(econ_data, output_dir / "economic_indicators.csv", [
        "country", "iso3", "year", "inflation_pct", "unemployment_pct",
        "debt_to_gdp_pct", "revenue_pct_gdp", "expenditure_pct_gdp",
        "current_account_pct_gdp", "fdi_inflow_usd", "fdi_outflow_usd", "reserves_usd"
    ])
    
    write_csv(energy_data, output_dir / "energy.csv", [
        "country", "iso3", "year", "energy_use_per_capita",
        "co2_emissions_per_capita", "renewable_energy_pct", "electricity_access_pct"
    ])

    # Write country metadata
    metadata = [{
        "iso3": c["iso3"], "name": c["name"], "region": c["region"],
        "subregion": c["subregion"], "is_core": c["is_core"]
    } for c in countries]
    write_csv(metadata, output_dir / "countries.csv", [
        "iso3", "name", "region", "subregion", "is_core"
    ])

    print(f"Written {len(gdp_data)} GDP rows")
    print(f"Written {len(pop_data)} population rows")
    print(f"Written {len(trade_data)} trade rows")
    print(f"Written {len(econ_data)} economic indicator rows")
    print(f"Written {len(energy_data)} energy rows")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()