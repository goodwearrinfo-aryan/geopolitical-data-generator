#!/usr/bin/env python3
"""Generate synthetic INSCR/CSP calibration fixtures.

Outputs CSV files matching INSCR data schemas for:
- Polity5 regime authority (polity2, regime_type)
- Coups d'Etat (successful, attempted, plotted)
- State Fragility Index (legitimacy, effectiveness, fragility)
- Major Episodes of Political Violence (MEPV)
- Forcibly Displaced Populations (refugees, IDPs)
- IGO Memberships

Usage:
    python tests/generate_inscr.py --countries 50 --years 10 --seed 42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from faker import Faker

CORE_COUNTRIES = [
    ("USA", "United States", "Americas", "Northern America", "democracy"),
    ("CHN", "China", "Asia", "Eastern Asia", "autocracy"),
    ("JPN", "Japan", "Asia", "Eastern Asia", "democracy"),
    ("DEU", "Germany", "Europe", "Western Europe", "democracy"),
    ("IND", "India", "Asia", "Southern Asia", "democracy"),
    ("GBR", "United Kingdom", "Europe", "Northern Europe", "democracy"),
    ("FRA", "France", "Europe", "Western Europe", "democracy"),
    ("ITA", "Italy", "Europe", "Southern Europe", "democracy"),
    ("CAN", "Canada", "Americas", "Northern America", "democracy"),
    ("KOR", "South Korea", "Asia", "Eastern Asia", "democracy"),
    ("RUS", "Russia", "Europe", "Eastern Europe", "autocracy"),
    ("BRA", "Brazil", "Americas", "South America", "democracy"),
    ("AUS", "Australia", "Oceania", "Australia and New Zealand", "democracy"),
    ("ESP", "Spain", "Europe", "Southern Europe", "democracy"),
    ("MEX", "Mexico", "Americas", "Central America", "democracy"),
    ("IDN", "Indonesia", "Asia", "South-Eastern Asia", "democracy"),
    ("NLD", "Netherlands", "Europe", "Western Europe", "democracy"),
    ("SAU", "Saudi Arabia", "Asia", "Western Asia", "autocracy"),
    ("TUR", "Turkey", "Asia", "Western Asia", "anocracy"),
    ("CHE", "Switzerland", "Europe", "Western Europe", "democracy"),
    ("TWN", "Taiwan", "Asia", "Eastern Asia", "democracy"),
    ("POL", "Poland", "Europe", "Eastern Europe", "democracy"),
    ("SWE", "Sweden", "Europe", "Northern Europe", "democracy"),
    ("BEL", "Belgium", "Europe", "Western Europe", "democracy"),
    ("THA", "Thailand", "Asia", "South-Eastern Asia", "anocracy"),
    ("IRN", "Iran", "Asia", "Southern Asia", "autocracy"),
    ("ARG", "Argentina", "Americas", "South America", "democracy"),
    ("NOR", "Norway", "Europe", "Northern Europe", "democracy"),
    ("NGA", "Nigeria", "Africa", "Western Africa", "anocracy"),
    ("ISR", "Israel", "Asia", "Western Asia", "democracy"),
    ("ARE", "UAE", "Asia", "Western Asia", "autocracy"),
    ("EGY", "Egypt", "Africa", "Northern Africa", "autocracy"),
    ("ZAF", "South Africa", "Africa", "Southern Africa", "democracy"),
    ("PAK", "Pakistan", "Asia", "Southern Asia", "anocracy"),
    ("BGD", "Bangladesh", "Asia", "Southern Asia", "anocracy"),
    ("PHL", "Philippines", "Asia", "South-Eastern Asia", "democracy"),
    ("VNM", "Vietnam", "Asia", "South-Eastern Asia", "autocracy"),
    ("COL", "Colombia", "Americas", "South America", "democracy"),
    ("MYS", "Malaysia", "Asia", "South-Eastern Asia", "anocracy"),
    ("SGP", "Singapore", "Asia", "South-Eastern Asia", "autocracy"),
    ("CHL", "Chile", "Americas", "South America", "democracy"),
    ("PER", "Peru", "Americas", "South America", "democracy"),
    ("FIN", "Finland", "Europe", "Northern Europe", "democracy"),
    ("CZE", "Czechia", "Europe", "Eastern Europe", "democracy"),
    ("ROU", "Romania", "Europe", "Eastern Europe", "democracy"),
    ("PRT", "Portugal", "Europe", "Southern Europe", "democracy"),
    ("GRC", "Greece", "Europe", "Southern Europe", "democracy"),
    ("HUN", "Hungary", "Europe", "Eastern Europe", "anocracy"),
    ("KAZ", "Kazakhstan", "Asia", "Central Asia", "autocracy"),
    ("IRQ", "Iraq", "Asia", "Western Asia", "anocracy"),
]

REGIME_POLITY_MAP = {
    "democracy": (6, 10),
    "anocracy": (-5, 5),
    "autocracy": (-10, -6),
    "failed_state": (-10, -8),
}


def generate_countries(n_countries: int, seed: int) -> List[Dict]:
    """Generate country metadata with base regime."""
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    countries = []
    for iso3, name, region, subregion, regime in CORE_COUNTRIES:
        if len(countries) >= n_countries:
            break
        countries.append({
            "iso3": iso3,
            "name": name,
            "region": region,
            "subregion": subregion,
            "base_regime": regime,
            "is_core": True,
        })

    return countries[:n_countries]


def generate_polity_data(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate Polity5 regime authority time series."""
    rows = []
    for c in countries:
        regime = c["base_regime"]
        min_pol, max_pol = REGIME_POLITY_MAP[regime]
        polity = rng.uniform(min_pol, max_pol)
        
        # Persistence parameter
        persistence = rng.uniform(0.85, 0.98)
        
        for year in years:
            # Polity evolves with persistence
            target = rng.uniform(min_pol, max_pol)
            polity = persistence * polity + (1 - persistence) * target + rng.normal(0, 0.5)
            polity = np.clip(polity, -10, 10)
            
            # Determine regime from polity
            if polity >= 6:
                regime_type = "democracy"
            elif polity <= -6:
                regime_type = "autocracy"
            elif polity >= -5 and polity <= 5:
                regime_type = "anocracy"
            else:
                regime_type = "failed_state"
            
            # Components
            democ = max(0, min(10, (polity + 10) / 2 + rng.normal(0, 0.5)))
            autoc = max(0, min(10, (10 - polity) / 2 + rng.normal(0, 0.5)))
            
            rows.append({
                "country": c["name"],
                "iso3": c["iso3"],
                "year": year,
                "polity2": round(polity, 1),
                "democ": round(democ, 1),
                "autoc": round(autoc, 1),
                "regime_type": regime_type,
                "durable": rng.integers(0, 50),
                "xrreg": rng.integers(0, 3),
                "xrcomp": rng.integers(0, 3),
                "xropen": rng.integers(0, 3),
            })
    return rows


def generate_coups_data(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate coup events (successful, attempted, plotted)."""
    rows = []
    coup_types = ["successful", "attempted", "plotted", "alleged"]
    
    for c in countries:
        # Base coup probability by regime
        regime = c["base_regime"]
        if regime == "autocracy":
            base_prob = 0.03
        elif regime == "anocracy":
            base_prob = 0.05
        elif regime == "failed_state":
            base_prob = 0.08
        else:
            base_prob = 0.005
        
        for year in years:
            if rng.random() < base_prob:
                coup_type = rng.choice(coup_types, p=[0.25, 0.4, 0.2, 0.15])
                success = coup_type == "successful"
                
                rows.append({
                    "country": c["name"],
                    "iso3": c["iso3"],
                    "year": year,
                    "coup_type": coup_type,
                    "success": int(success),
                    "leader_removed": int(success and rng.random() < 0.7),
                    "constitution_suspended": int(success and rng.random() < 0.5),
                    "days_duration": rng.integers(1, 30) if not success else rng.integers(1, 365),
                    "fatalities": rng.integers(0, 1000) if success else rng.integers(0, 100),
                })
    return rows


def generate_state_fragility(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate State Fragility Index (legitimacy, effectiveness, total)."""
    rows = []
    for c in countries:
        regime = c["base_regime"]
        
        # Base fragility by regime
        if regime == "democracy":
            base_legitimacy = rng.uniform(2, 6)
            base_effectiveness = rng.uniform(2, 6)
        elif regime == "anocracy":
            base_legitimacy = rng.uniform(5, 12)
            base_effectiveness = rng.uniform(5, 12)
        elif regime == "autocracy":
            base_legitimacy = rng.uniform(8, 15)
            base_effectiveness = rng.uniform(5, 12)
        else:  # failed_state
            base_legitimacy = rng.uniform(15, 25)
            base_effectiveness = rng.uniform(15, 25)
        
        for year in years:
            # Add noise and trend
            legitimacy = np.clip(base_legitimacy + rng.normal(0, 1.5) + rng.uniform(-0.5, 0.5), 0, 25)
            effectiveness = np.clip(base_effectiveness + rng.normal(0, 1.5) + rng.uniform(-0.5, 0.5), 0, 25)
            fragility = legitimacy + effectiveness
            
            # Components (4 each for legitimacy and effectiveness)
            sec_leg = np.clip(legitimacy / 4 + rng.normal(0, 1), 0, 6.25)
            pol_leg = np.clip(legitimacy / 4 + rng.normal(0, 1), 0, 6.25)
            eco_leg = np.clip(legitimacy / 4 + rng.normal(0, 1), 0, 6.25)
            soc_leg = np.clip(legitimacy / 4 + rng.normal(0, 1), 0, 6.25)
            
            sec_eff = np.clip(effectiveness / 4 + rng.normal(0, 1), 0, 6.25)
            pol_eff = np.clip(effectiveness / 4 + rng.normal(0, 1), 0, 6.25)
            eco_eff = np.clip(effectiveness / 4 + rng.normal(0, 1), 0, 6.25)
            soc_eff = np.clip(effectiveness / 4 + rng.normal(0, 1), 0, 6.25)
            
            rows.append({
                "country": c["name"],
                "iso3": c["iso3"],
                "year": year,
                "legitimacy_index": round(legitimacy, 2),
                "effectiveness_index": round(effectiveness, 2),
                "fragility_index": round(fragility, 2),
                "security_legitimacy": round(sec_leg, 2),
                "political_legitimacy": round(pol_leg, 2),
                "economic_legitimacy": round(eco_leg, 2),
                "social_legitimacy": round(soc_leg, 2),
                "security_effectiveness": round(sec_eff, 2),
                "political_effectiveness": round(pol_eff, 2),
                "economic_effectiveness": round(eco_eff, 2),
                "social_effectiveness": round(soc_eff, 2),
            })
    return rows


def generate_mepv(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate Major Episodes of Political Violence (MEPV)."""
    rows = []
    violence_types = ["interstate", "civil", "ethnic", "communal"]
    
    for c in countries:
        regime = c["base_regime"]
        # Base conflict probability by regime
        if regime == "failed_state":
            base_prob = 0.15
        elif regime == "anocracy":
            base_prob = 0.08
        elif regime == "autocracy":
            base_prob = 0.04
        else:
            base_prob = 0.01
        
        for year in years:
            if rng.random() < base_prob:
                vtype = rng.choice(violence_types, p=[0.1, 0.5, 0.3, 0.1])
                
                # Magnitude scores (0-10)
                if vtype == "interstate":
                    magnitude = rng.uniform(3, 10)
                elif vtype == "civil":
                    magnitude = rng.uniform(2, 8)
                elif vtype == "ethnic":
                    magnitude = rng.uniform(2, 7)
                else:
                    magnitude = rng.uniform(1, 5)
                
                # Fatalities (log scale)
                fatalities = int(10 ** rng.uniform(2, 5)) if magnitude > 5 else int(10 ** rng.uniform(1, 3))
                
                start_year = year
                end_year = year + rng.integers(0, 5)
                
                rows.append({
                    "country": c["name"],
                    "iso3": c["iso3"],
                    "start_year": start_year,
                    "end_year": end_year,
                    "violence_type": vtype,
                    "magnitude_score": round(magnitude, 1),
                    "fatalities": fatalities,
                    "area_affected_pct": round(rng.uniform(1, 50), 1),
                    "refugees_generated": int(fatalities * rng.uniform(0.5, 5)),
                    "ongoing": int(end_year >= years[-1]),
                })
    return rows


def generate_displaced_populations(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate forcibly displaced populations (refugees, IDPs)."""
    rows = []
    for c in countries:
        regime = c["base_regime"]
        base_refugees = {"democracy": 1000, "anocracy": 10000, "autocracy": 50000, "failed_state": 200000}[regime]
        base_idps = {"democracy": 5000, "anocracy": 50000, "autocracy": 100000, "failed_state": 500000}[regime]
        
        refugees = base_refugees
        idps = base_idps
        
        for year in years:
            # Evolve with conflict
            shock = rng.lognormal(0, 0.5)
            refugees = int(refugees * shock * rng.uniform(0.8, 1.3))
            idps = int(idps * shock * rng.uniform(0.8, 1.3))
            
            # Source and host
            source_refugees = int(refugees * rng.uniform(0.3, 0.8))
            host_refugees = int(refugees * rng.uniform(0.2, 0.5))
            
            rows.append({
                "country": c["name"],
                "iso3": c["iso3"],
                "year": year,
                "refugees_total": refugees,
                "refugees_source": source_refugees,
                "refugees_host": host_refugees,
                "idps_total": idps,
                "asylum_seekers": int(refugees * rng.uniform(0.05, 0.2)),
                "stateless_persons": int(refugees * rng.uniform(0.01, 0.05)),
                "returned_refugees": int(refugees * rng.uniform(0.01, 0.1)),
                "returned_idps": int(idps * rng.uniform(0.01, 0.1)),
            })
    return rows


def generate_igo_memberships(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate IGO memberships."""
    rows = []
    igo_types = [
        ("universal", 39),      # Universal membership IGOs
        ("intercontinental", 52), # Inter-continental
        ("regional", 288),      # Regionally-defined
    ]
    
    for c in countries:
        for year in years:
            for igo_type, max_igos in igo_types:
                # Probability of membership varies by type and country size
                if igo_type == "universal":
                    prob = 0.9
                elif igo_type == "intercontinental":
                    prob = 0.6 if c["is_core"] else 0.4
                else:
                    prob = 0.3 if c["is_core"] else 0.2
                
                n_members = rng.binomial(max_igos, prob)
                
                rows.append({
                    "country": c["name"],
                    "iso3": c["iso3"],
                    "year": year,
                    "igo_type": igo_type,
                    "memberships": n_members,
                    "max_possible": max_igos,
                })
    return rows


def write_csv(rows: List[Dict], filepath: Path, fieldnames: List[str]):
    """Write rows to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic INSCR/CSP calibration fixtures")
    parser.add_argument("--countries", type=int, default=50, help="Number of countries")
    parser.add_argument("--years", type=int, default=10, help="Number of years of history")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year")
    parser.add_argument("--output", type=str, default="tests/fixtures/inscr/v1", help="Output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    years = list(range(args.start_year, args.start_year + args.years))
    countries = generate_countries(args.countries, args.seed)
    output_dir = Path(args.output)

    print(f"Generating INSCR fixtures for {len(countries)} countries, {len(years)} years ({years[0]}-{years[-1]})")

    # Generate all datasets
    polity_data = generate_polity_data(countries, years, rng)
    coups_data = generate_coups_data(countries, years, rng)
    fragility_data = generate_state_fragility(countries, years, rng)
    mepv_data = generate_mepv(countries, years, rng)
    displaced_data = generate_displaced_populations(countries, years, rng)
    igo_data = generate_igo_memberships(countries, years, rng)

    # Write CSVs
    write_csv(polity_data, output_dir / "polity.csv", [
        "country", "iso3", "year", "polity2", "democ", "autoc",
        "regime_type", "durable", "xrreg", "xrcomp", "xropen"
    ])
    
    write_csv(coups_data, output_dir / "coups.csv", [
        "country", "iso3", "year", "coup_type", "success",
        "leader_removed", "constitution_suspended", "days_duration", "fatalities"
    ])
    
    write_csv(fragility_data, output_dir / "state_fragility.csv", [
        "country", "iso3", "year", "legitimacy_index", "effectiveness_index",
        "fragility_index", "security_legitimacy", "political_legitimacy",
        "economic_legitimacy", "social_legitimacy", "security_effectiveness",
        "political_effectiveness", "economic_effectiveness", "social_effectiveness"
    ])
    
    write_csv(mepv_data, output_dir / "mepv.csv", [
        "country", "iso3", "start_year", "end_year", "violence_type",
        "magnitude_score", "fatalities", "area_affected_pct", "refugees_generated", "ongoing"
    ])
    
    write_csv(displaced_data, output_dir / "displaced_populations.csv", [
        "country", "iso3", "year", "refugees_total", "refugees_source",
        "refugees_host", "idps_total", "asylum_seekers", "stateless_persons",
        "returned_refugees", "returned_idps"
    ])
    
    write_csv(igo_data, output_dir / "igo_memberships.csv", [
        "country", "iso3", "year", "igo_type", "memberships", "max_possible"
    ])

    print(f"Written {len(polity_data)} polity rows")
    print(f"Written {len(coups_data)} coup rows")
    print(f"Written {len(fragility_data)} fragility rows")
    print(f"Written {len(mepv_data)} MEPV rows")
    print(f"Written {len(displaced_data)} displaced population rows")
    print(f"Written {len(igo_data)} IGO membership rows")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()