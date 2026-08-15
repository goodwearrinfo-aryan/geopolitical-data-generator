#!/usr/bin/env python3
"""Generate synthetic ACLED conflict event fixtures.

Outputs CSV matching ACLED event schema for:
- Battles, explosions/remote violence, violence against civilians
- Protests, riots, strategic developments
- With fatalities, actors, locations, dates

Usage:
    python tests/generate_acled.py --countries 50 --years 10 --seed 42
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
from faker import Faker

CORE_COUNTRIES = [
    ("USA", "United States", "Americas", "Northern America", 39.8, -98.6),
    ("CHN", "China", "Asia", "Eastern Asia", 35.9, 104.2),
    ("JPN", "Japan", "Asia", "Eastern Asia", 36.2, 138.3),
    ("DEU", "Germany", "Europe", "Western Europe", 51.2, 10.5),
    ("IND", "India", "Asia", "Southern Asia", 20.6, 78.9),
    ("GBR", "United Kingdom", "Europe", "Northern Europe", 55.4, -3.4),
    ("FRA", "France", "Europe", "Western Europe", 46.2, 2.2),
    ("ITA", "Italy", "Europe", "Southern Europe", 41.9, 12.6),
    ("CAN", "Canada", "Americas", "Northern America", 56.1, -106.3),
    ("KOR", "South Korea", "Asia", "Eastern Asia", 35.9, 127.8),
    ("RUS", "Russia", "Europe", "Eastern Europe", 61.5, 105.3),
    ("BRA", "Brazil", "Americas", "South America", -14.2, -51.9),
    ("AUS", "Australia", "Oceania", "Australia and New Zealand", -25.3, 133.8),
    ("ESP", "Spain", "Europe", "Southern Europe", 40.5, -3.7),
    ("MEX", "Mexico", "Americas", "Central America", 23.6, -102.6),
    ("IDN", "Indonesia", "Asia", "South-Eastern Asia", -0.8, 113.9),
    ("NLD", "Netherlands", "Europe", "Western Europe", 52.1, 5.3),
    ("SAU", "Saudi Arabia", "Asia", "Western Asia", 23.9, 45.1),
    ("TUR", "Turkey", "Asia", "Western Asia", 38.9, 35.2),
    ("CHE", "Switzerland", "Europe", "Western Europe", 46.8, 8.2),
    ("TWN", "Taiwan", "Asia", "Eastern Asia", 23.7, 120.9),
    ("POL", "Poland", "Europe", "Eastern Europe", 51.9, 19.1),
    ("SWE", "Sweden", "Europe", "Northern Europe", 60.1, 18.6),
    ("BEL", "Belgium", "Europe", "Western Europe", 50.5, 4.5),
    ("THA", "Thailand", "Asia", "South-Eastern Asia", 15.9, 101.0),
    ("IRN", "Iran", "Asia", "Southern Asia", 32.4, 53.7),
    ("ARG", "Argentina", "Americas", "South America", -38.4, -63.6),
    ("NOR", "Norway", "Europe", "Northern Europe", 60.5, 8.5),
    ("NGA", "Nigeria", "Africa", "Western Africa", 9.1, 8.7),
    ("ISR", "Israel", "Asia", "Western Asia", 31.0, 34.9),
    ("ARE", "UAE", "Asia", "Western Asia", 23.4, 53.8),
    ("EGY", "Egypt", "Africa", "Northern Africa", 26.8, 30.8),
    ("ZAF", "South Africa", "Africa", "Southern Africa", -30.6, 22.9),
    ("PAK", "Pakistan", "Asia", "Southern Asia", 30.4, 69.3),
    ("BGD", "Bangladesh", "Asia", "Southern Asia", 23.7, 90.4),
    ("PHL", "Philippines", "Asia", "South-Eastern Asia", 12.9, 121.8),
    ("VNM", "Vietnam", "Asia", "South-Eastern Asia", 14.1, 108.3),
    ("COL", "Colombia", "Americas", "South America", 4.6, -74.1),
    ("MYS", "Malaysia", "Asia", "South-Eastern Asia", 4.2, 101.9),
    ("SGP", "Singapore", "Asia", "South-Eastern Asia", 1.4, 103.8),
    ("CHL", "Chile", "Americas", "South America", -35.7, -71.5),
    ("PER", "Peru", "Americas", "South America", -9.2, -75.0),
    ("FIN", "Finland", "Europe", "Northern Europe", 61.9, 25.7),
    ("CZE", "Czechia", "Europe", "Eastern Europe", 49.8, 15.5),
    ("ROU", "Romania", "Europe", "Eastern Europe", 45.9, 25.0),
    ("PRT", "Portugal", "Europe", "Southern Europe", 39.4, -8.2),
    ("GRC", "Greece", "Europe", "Southern Europe", 39.1, 21.8),
    ("HUN", "Hungary", "Europe", "Eastern Europe", 47.2, 19.5),
    ("KAZ", "Kazakhstan", "Asia", "Central Asia", 48.0, 66.9),
    ("IRQ", "Iraq", "Asia", "Western Asia", 33.2, 43.7),
]

EVENT_TYPES = {
    "battle": ["armed clash", "government regains territory", "non-state actor overtakes territory"],
    "explosions_remote_violence": ["air/drone strike", "shelling/artillery/missile attack", "remote explosive/landmine/IED", "grenade"],
    "violence_against_civilians": ["attack", "sexual violence", "abduction/forced disappearance"],
    "protests": ["peaceful protest", "protest with intervention", "excessive force against protesters"],
    "riots": ["riot", "mob violence"],
    "strategic_developments": ["agreement", "arrest", "change to group/activity", "headquarters/base established", "looting/property destruction", "non-violent transfer of territory"],
}

ACTORS = {
    "state": ["Government", "Military", "Police", "State Security Forces", "Pro-Government Militia"],
    "rebel": ["Rebels", "Opposition Forces", "Insurgents", "Separatists", "Jihadists"],
    "civilian": ["Civilians", "Protesters", "Rioters", "Voters", "Refugees/IDPs"],
    "external": ["External Forces", "Foreign Military", "Peacekeepers", "UN Forces"],
    "other": ["Private Security", "Vigilantes", "Criminal Group", "Gang", "Militia"],
}


def generate_countries(n_countries: int, seed: int) -> List[Dict]:
    """Generate country metadata with coordinates."""
    rng = np.random.default_rng(seed)
    countries = []
    for iso3, name, region, subregion, lat, lon in CORE_COUNTRIES:
        if len(countries) >= n_countries:
            break
        countries.append({
            "iso3": iso3,
            "name": name,
            "region": region,
            "subregion": subregion,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "is_core": True,
        })
    return countries[:n_countries]


def generate_events(countries: List[Dict], years: List[int], rng: np.random.Generator) -> List[Dict]:
    """Generate ACLED-style conflict events."""
    rows = []
    event_id = 1
    
    for c in countries:
        # Base event rate by region/regime (simplified)
        if c["region"] in ["Africa", "Middle East"]:
            base_daily_rate = rng.uniform(0.5, 3.0)
        elif c["region"] in ["Asia", "South America"]:
            base_daily_rate = rng.uniform(0.2, 1.5)
        else:
            base_daily_rate = rng.uniform(0.05, 0.5)
        
        for year in years:
            days_in_year = 366 if (year % 4 == 0) else 365
            n_events = max(1, int(base_daily_rate * days_in_year * rng.uniform(0.5, 2.0)))
            
            # Generate events across the year
            for _ in range(n_events):
                day_of_year = int(rng.integers(1, days_in_year + 1))
                event_date = date(year, 1, 1) + timedelta(days=day_of_year - 1)
                
                # Event type
                event_type = rng.choice(list(EVENT_TYPES.keys()), p=[0.25, 0.15, 0.15, 0.2, 0.1, 0.15])
                sub_event_type = rng.choice(EVENT_TYPES[event_type])
                
                # Actors
                actor1_type = rng.choice(list(ACTORS.keys()), p=[0.35, 0.25, 0.15, 0.1, 0.15])
                actor1 = rng.choice(ACTORS[actor1_type])
                
                actor2_type = rng.choice(list(ACTORS.keys()), p=[0.2, 0.3, 0.2, 0.1, 0.2])
                actor2 = rng.choice(ACTORS[actor2_type])
                
                # Interaction code
                interaction = f"{actor1_type[0]}{actor2_type[0]}"  # e.g., "sr" for state-rebel
                
                # Location (perturb around centroid)
                lat = c["centroid_lat"] + rng.uniform(-5, 5)
                lon = c["centroid_lon"] + rng.uniform(-5, 5)
                lat = np.clip(lat, -90, 90)
                lon = np.clip(lon, -180, 180)
                
                # Admin boundaries (simplified)
                admin1 = f"{c['iso3']}_ADM1_{rng.integers(1, 20)}"
                admin2 = f"{admin1}_ADM2_{rng.integers(1, 10)}"
                admin3 = f"{admin2}_ADM3_{rng.integers(1, 5)}"
                
                # Fatalities
                if event_type == "battle":
                    fatalities = int(rng.lognormal(2, 1.5))
                elif event_type == "explosions_remote_violence":
                    fatalities = int(rng.lognormal(1.5, 1.2))
                elif event_type == "violence_against_civilians":
                    fatalities = int(rng.lognormal(1, 1))
                elif event_type == "riots":
                    fatalities = int(rng.lognormal(0.5, 0.8))
                else:
                    fatalities = int(rng.lognormal(0, 0.5))
                
                fatalities = min(fatalities, 1000)
                
                # Civilian targeting
                civilian_targeting = int(event_type == "violence_against_civilians" or 
                                        (event_type in ["battle", "explosions_remote_violence"] and rng.random() < 0.15))
                
                rows.append({
                    "event_id": event_id,
                    "event_date": event_date.isoformat(),
                    "year": year,
                    "time_precision": 1,  # exact date
                    "event_type": event_type,
                    "sub_event_type": sub_event_type,
                    "actor1": actor1,
                    "assoc_actor_1": "",
                    "inter1": actor1_type[0],
                    "actor2": actor2,
                    "assoc_actor_2": "",
                    "inter2": actor2_type[0],
                    "interaction": interaction,
                    "country": c["name"],
                    "iso3": c["iso3"],
                    "region": c["region"],
                    "admin1": admin1,
                    "admin2": admin2,
                    "admin3": admin3,
                    "location": f"{c['name']} ({admin1})",
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "geo_precision": 1,  # exact coordinates
                    "source": "synthetic",
                    "source_scale": "local",
                    "notes": "",
                    "fatalities": fatalities,
                    "civilian_targeting": civilian_targeting,
                    "timestamp": f"{event_date.isoformat()}T{rng.integers(0,24):02d}:{rng.integers(0,60):02d}:00Z",
                })
                event_id += 1
    
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
    parser = argparse.ArgumentParser(description="Generate synthetic ACLED conflict event fixtures")
    parser.add_argument("--countries", type=int, default=50, help="Number of countries")
    parser.add_argument("--years", type=int, default=10, help="Number of years of history")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year")
    parser.add_argument("--output", type=str, default="tests/fixtures/acled/v1", help="Output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    years = list(range(args.start_year, args.start_year + args.years))
    countries = generate_countries(args.countries, args.seed)
    output_dir = Path(args.output)

    print(f"Generating ACLED fixtures for {len(countries)} countries, {len(years)} years ({years[0]}-{years[-1]})")

    events = generate_events(countries, years, rng)

    # Write main events file
    fieldnames = [
        "event_id", "event_date", "year", "time_precision", "event_type",
        "sub_event_type", "actor1", "assoc_actor_1", "inter1", "actor2",
        "assoc_actor_2", "inter2", "interaction", "country", "iso3",
        "region", "admin1", "admin2", "admin3", "location", "latitude",
        "longitude", "geo_precision", "source", "source_scale", "notes",
        "fatalities", "civilian_targeting", "timestamp"
    ]
    write_csv(events, output_dir / "events.csv", fieldnames)

    # Also write actor dictionary
    actor_dict = []
    for cat, actors in ACTORS.items():
        for actor in actors:
            actor_dict.append({"actor_name": actor, "actor_category": cat})
    write_csv(actor_dict, output_dir / "actors.csv", ["actor_name", "actor_category"])

    # Write event type dictionary
    event_dict = []
    for etype, subtypes in EVENT_TYPES.items():
        for subtype in subtypes:
            event_dict.append({"event_type": etype, "sub_event_type": subtype})
    write_csv(event_dict, output_dir / "event_types.csv", ["event_type", "sub_event_type"])

    print(f"Written {len(events)} events")
    print(f"Written {len(actor_dict)} actor definitions")
    print(f"Written {len(event_dict)} event type definitions")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()