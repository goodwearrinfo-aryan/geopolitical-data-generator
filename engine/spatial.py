"""Spatial engine for geographic relationships, trade gravity, and contagion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from schemas.core import SimulationState, Country, ResourceType


@dataclass
class CountryGeometry:
    """Geometric properties of a country."""
    iso3: str
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    neighbors: List[str]  # ISO3 codes of adjacent countries
    coastline_km: float = 0.0
    capital_lat: Optional[float] = None
    capital_lon: Optional[float] = None


class SpatialEngine:
    """Manages spatial relationships and geographic calculations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.geometries: Dict[str, CountryGeometry] = {}
        self.distance_matrix: Dict[Tuple[str, str], float] = {}
        self.trade_graph: nx.DiGraph = nx.DiGraph()
        self.contagion_graph: nx.Graph = nx.Graph()
        self._load_geometries()
        self._compute_distances()
    
    def _load_geometries(self) -> None:
        """Load country geometries from Natural Earth data or use defaults."""
        # Default centroid coordinates for major countries
        # In production, load from GeoJSON
        default_coords = {
            "USA": (39.8, -98.6), "CHN": (35.9, 104.2), "RUS": (61.5, 105.3),
            "IND": (20.6, 78.9), "BRA": (-14.2, -51.9), "CAN": (56.1, -106.3),
            "AUS": (-25.3, 133.8), "DEU": (51.2, 10.5), "FRA": (46.2, 2.2),
            "GBR": (55.4, -3.4), "JPN": (36.2, 138.3), "KOR": (35.9, 127.8),
            "MEX": (23.6, -102.6), "IDN": (-0.8, 113.9), "SAU": (23.9, 45.1),
            "TUR": (38.9, 35.2), "IRN": (32.4, 53.7), "EGY": (26.8, 30.8),
            "NGA": (9.1, 8.7), "ZAF": (-30.6, 22.9), "ARG": (-38.4, -63.6),
            "PAK": (30.4, 69.3), "BGD": (23.7, 90.4), "VNM": (14.1, 108.3),
            "PHL": (12.9, 121.8), "THA": (15.9, 101.0), "MYS": (4.2, 101.9),
            "POL": (51.9, 19.1), "UKR": (48.4, 31.2), "ROU": (45.9, 25.0),
            "KAZ": (48.0, 66.9), "UZB": (41.4, 64.6), "VEN": (6.4, -66.6),
            "COL": (4.6, -74.1), "CHL": (-35.7, -71.5), "PER": (-9.2, -75.0),
            "IRQ": (33.2, 43.7), "AFG": (33.9, 67.7), "SYR": (34.8, 38.9),
            "YEM": (15.6, 48.5), "ISR": (31.0, 34.9), "JOR": (31.2, 36.2),
            "LBN": (33.9, 35.5), "OMN": (21.5, 55.9), "QAT": (25.4, 51.2),
            "KWT": (29.3, 47.5), "BHR": (26.0, 50.6), "ARE": (23.4, 53.8),
            "SGP": (1.4, 103.8), "HKG": (22.3, 114.2), "TWN": (23.7, 120.9),
            "NZL": (-40.9, 174.9), "NOR": (60.5, 8.5), "SWE": (60.1, 18.6),
            "FIN": (61.9, 25.7), "DNK": (56.3, 9.5), "AUT": (47.5, 14.6),
            "CHE": (46.8, 8.2), "BEL": (50.5, 4.5), "NLD": (52.1, 5.3),
            "IRL": (53.4, -8.2), "PRT": (39.4, -8.2), "GRC": (39.1, 21.8),
            "CZE": (49.8, 15.5), "HUN": (47.2, 19.5), "SVK": (48.7, 19.7),
            "BGR": (42.7, 25.5), "HRV": (45.1, 15.2), "SRB": (44.0, 21.0),
            "BIH": (43.9, 17.7), "ALB": (41.2, 20.2), "MKD": (41.6, 21.7),
            "MNE": (42.7, 19.3), "SVN": (46.2, 14.8), "LTU": (55.2, 23.9),
            "LVA": (56.9, 24.6), "EST": (58.6, 25.0), "MDA": (47.0, 28.8),
            "GEO": (42.3, 43.4), "ARM": (40.1, 45.0), "AZE": (40.1, 47.6),
            "BLR": (53.7, 27.9), "MNG": (46.9, 103.8), "KGZ": (41.2, 74.8),
            "TJK": (38.9, 71.3), "TKM": (38.9, 59.6), "NPL": (28.4, 84.1),
            "BTN": (27.5, 90.4), "LKA": (7.9, 80.8), "MDV": (3.2, 73.2),
            "LAO": (19.9, 102.5), "MMR": (21.9, 95.9), "KHM": (12.6, 105.0),
            "BRN": (4.5, 114.7), "TLS": (-8.9, 125.7), "PNG": (-6.3, 143.9),
            "SLB": (-9.6, 160.2), "VUT": (-15.4, 166.9), "FJI": (-17.7, 178.1),
            "WSM": (-13.8, -171.8), "TON": (-21.2, -175.2), "KIR": (1.9, -157.4),
            "FSM": (7.4, 150.5), "MHL": (7.1, 171.2), "PLW": (7.5, 134.6),
            "NRU": (-0.5, 166.9), "TUV": (-7.1, 178.3),
            "ETH": (9.1, 40.5), "KEN": (-0.0, 37.9), "TZA": (-6.4, 34.8),
            "UGA": (1.4, 32.3), "RWA": (-1.9, 29.9), "BDI": (-3.4, 29.9),
            "COD": (-4.0, 21.8), "COG": (-0.2, 14.9), "GAB": (-0.8, 11.6),
            "CMR": (7.4, 12.4), "CAF": (6.6, 20.9), "TCD": (15.5, 18.7),
            "SDN": (12.9, 30.2), "SSD": (6.9, 31.3), "ERI": (15.2, 39.8),
            "DJI": (11.8, 42.6), "SOM": (5.2, 46.2), "MLI": (17.6, -3.9),
            "MRT": (21.0, -10.9), "SEN": (14.5, -14.5), "GMB": (13.4, -15.3),
            "GNB": (11.8, -15.1), "GIN": (9.9, -9.7), "SLE": (8.5, -11.8),
            "LBR": (6.4, -9.4), "CIV": (7.5, -5.5), "GHA": (7.9, -1.0),
            "TGO": (8.6, 0.8), "BEN": (9.3, 2.3), "BFA": (12.2, -1.6),
            "NER": (17.6, 8.1), "ESH": (24.2, -12.8), "MAR": (31.8, -7.1),
            "DZA": (28.0, 1.7), "TUN": (33.9, 9.5), "LBY": (26.3, 17.2),
            "CUB": (21.5, -77.8), "HTI": (19.0, -72.4), "DOM": (18.7, -70.2),
            "JAM": (18.1, -77.3), "BHS": (25.0, -77.3), "TTO": (10.7, -61.2),
            "BRB": (13.2, -59.6), "LCA": (13.9, -60.9), "VCT": (13.2, -61.2),
            "GRD": (12.1, -61.7), "ATG": (17.1, -61.8), "KNA": (17.3, -62.7),
            "DMA": (15.4, -61.4), "VUT": (-15.4, 166.9), "WSM": (-13.8, -171.8),
            "TON": (-21.2, -175.2), "KIR": (1.9, -157.4), "FSM": (7.4, 150.5),
            "MHL": (7.1, 171.2), "PLW": (7.5, 134.6), "NRU": (-0.5, 166.9),
            "TUV": (-7.1, 178.3), "XKX": (42.6, 20.9),
        }
        
        # Neighbor relationships (simplified - in production load from topology)
        neighbors = self._get_default_neighbors()
        
        for iso3, (lat, lon) in default_coords.items():
            self.geometries[iso3] = CountryGeometry(
                iso3=iso3,
                centroid_lat=lat,
                centroid_lon=lon,
                area_km2=0,  # Will be filled from country data
                neighbors=neighbors.get(iso3, []),
            )
    
    def _get_default_neighbors(self) -> Dict[str, List[str]]:
        """Return default neighbor relationships."""
        # Simplified - real implementation would load from topology
        return {
            "USA": ["CAN", "MEX"], "CAN": ["USA"], "MEX": ["USA", "GTM", "BLZ"],
            "GTM": ["MEX", "BLZ", "SLV", "HND"], "BLZ": ["MEX", "GTM"],
            "SLV": ["GTM", "HND"], "HND": ["GTM", "SLV", "NIC"],
            "NIC": ["HND", "CRI"], "CRI": ["NIC", "PAN"], "PAN": ["CRI", "COL"],
            "COL": ["PAN", "VEN", "BRA", "PER", "ECU"], "VEN": ["COL", "BRA", "GUY"],
            "GUY": ["VEN", "BRA", "SUR"], "SUR": ["GUY", "BRA", "GUF"],
            "GUF": ["SUR", "BRA"], "BRA": ["GUF", "SUR", "GUY", "VEN", "COL", "PER", "BOL", "PRY", "ARG", "URY"],
            "PRY": ["BRA", "BOL", "ARG"], "BOL": ["BRA", "PRY", "ARG", "CHL", "PER"],
            "CHL": ["PER", "BOL", "ARG"], "ARG": ["CHL", "BOL", "PRY", "BRA", "URY"],
            "URY": ["BRA", "ARG"], "ECU": ["COL", "PER"], "PER": ["ECU", "COL", "BRA", "BOL", "CHL"],
            "DEU": ["DNK", "POL", "CZE", "AUT", "CHE", "FRA", "BEL", "NLD", "LUX"],
            "FRA": ["BEL", "LUX", "DEU", "CHE", "ITA", "ESP", "AND", "MCO"],
            "ESP": ["FRA", "AND", "PRT", "GBR"], "PRT": ["ESP"],
            "ITA": ["FRA", "CHE", "AUT", "SVN", "VAT", "SMR"],
            "CHE": ["DEU", "FRA", "ITA", "AUT", "LIE"],
            "AUT": ["DEU", "CZE", "SVK", "HUN", "SVN", "ITA", "CHE", "LIE"],
            "POL": ["DEU", "CZE", "SVK", "UKR", "BLR", "LTU", "RUS"],
            "CZE": ["DEU", "POL", "SVK", "AUT"],
            "SVK": ["CZE", "POL", "UKR", "HUN", "AUT"],
            "HUN": ["AUT", "SVK", "UKR", "ROU", "SRB", "HRV", "SVN"],
            "ROU": ["HUN", "SRB", "BGR", "MDA", "UKR"],
            "BGR": ["ROU", "SRB", "MKD", "GRC", "TUR"],
            "GRC": ["BGR", "MKD", "ALB", "TUR"],
            "TUR": ["GRC", "BGR", "GEO", "ARM", "AZE", "IRN", "IRQ", "SYR"],
            "SYR": ["TUR", "IRQ", "JOR", "ISR", "LBN"],
            "IRQ": ["TUR", "SYR", "JOR", "SAU", "KWT", "IRN"],
            "IRN": ["TUR", "IRQ", "ARM", "AZE", "TKM", "AFG", "PAK"],
            "SAU": ["IRQ", "JOR", "KWT", "QAT", "ARE", "OMN", "YEM"],
            "YEM": ["SAU", "OMN"], "OMN": ["YEM", "SAU", "ARE"],
            "ARE": ["SAU", "OMN", "QAT"], "QAT": ["SAU", "ARE"],
            "KWT": ["SAU", "IRQ"], "BHR": ["QAT", "SAU"],
            "JOR": ["SYR", "IRQ", "SAU", "ISR", "PSE"],
            "ISR": ["JOR", "SYR", "LBN", "EGY", "PSE"],
            "LBN": ["SYR", "ISR"], "EGY": ["ISR", "LBY", "SDN"],
            "LBY": ["EGY", "SDN", "TCD", "NER", "DZA", "TUN"],
            "TUN": ["DZA", "LBY"], "DZA": ["TUN", "LBY", "NER", "MLI", "MRT", "MAR", "ESH"],
            "MAR": ["DZA", "ESH"], "ESH": ["MAR", "DZA", "MRT"],
            "MRT": ["ESH", "DZA", "MLI", "SEN"],
            "MLI": ["MRT", "DZA", "NER", "BFA", "CIV", "GIN", "SEN", "MRT"],
            "NER": ["DZA", "LBY", "TCD", "NGA", "BEN", "BFA", "MLI"],
            "TCD": ["LBY", "SDN", "CAF", "CMR", "NER", "NER"],
            "SDN": ["EGY", "LBY", "TCD", "CAF", "SSD", "ERI", "ETH"],
            "SSD": ["SDN", "CAF", "COD", "UGA", "KEN", "ETH"],
            "ETH": ["ERI", "DJI", "SOM", "KEN", "SSD", "SDN"],
            "ERI": ["SDN", "ETH", "DJI"], "DJI": ["ERI", "ETH", "SOM"],
            "SOM": ["DJI", "ETH", "KEN"], "KEN": ["SOM", "ETH", "SSD", "UGA", "TZA"],
            "TZA": ["KEN", "UGA", "RWA", "BDI", "COD", "ZMB", "MWI", "MOZ"],
            "UGA": ["SSD", "KEN", "TZA", "RWA", "COD"],
            "RWA": ["UGA", "TZA", "BDI", "COD"],
            "BDI": ["RWA", "TZA", "COD"],
            "COD": ["SSD", "UGA", "RWA", "BDI", "TZA", "ZMB", "AGO", "COG", "CAF", "SDN"],
            "COG": ["COD", "CMR", "CAF", "GAB", "CAB"],
            "GAB": ["COG", "CMR", "GNQ"], "GNQ": ["GAB", "CMR"],
            "CMR": ["NGA", "TCD", "CAF", "COG", "GAB", "GNQ"],
            "NGA": ["BEN", "NER", "TCD", "CMR"],
            "BEN": ["NGA", "NER", "BFA", "TGO"],
            "TGO": ["BEN", "BFA", "GHA"],
            "GHA": ["TGO", "BFA", "CIV"],
            "CIV": ["GHA", "BFA", "MLI", "GIN", "LBR"],
            "LBR": ["CIV", "GIN", "SLE"],
            "SLE": ["LBR", "GIN"],
            "GIN": ["CIV", "LBR", "SLE", "GNB", "SEN", "MLI"],
            "GNB": ["GIN", "SEN"], "SEN": ["GNB", "GIN", "MLI", "MRT"],
            "CPV": [], "STP": [], "COM": [], "MDG": [], "SYC": [], "MUS": [],
            "JPN": [], "KOR": ["PRK"], "PRK": ["KOR", "CHN", "RUS"],
            "MNG": ["CHN", "RUS"], "CHN": ["MNG", "RUS", "PRK", "KOR", "HKG", "MAC", "VNM", "LAO", "MMR", "BTN", "NPL", "IND", "PAK", "AFG", "TKM", "UZB", "KGZ", "TJK", "KAZ"],
            "KAZ": ["CHN", "RUS", "KGZ", "UZB", "TKM"],
            "KGZ": ["CHN", "KAZ", "TJK", "UZB"],
            "UZB": ["KAZ", "KGZ", "TJK", "AFG", "TKM"],
            "TJK": ["CHN", "KGZ", "UZB", "AFG"],
            "TKM": ["KAZ", "UZB", "AFG", "IRN"],
            "AFG": ["TKM", "UZB", "TJK", "CHN", "PAK", "IRN"],
            "PAK": ["AFG", "CHN", "IND", "IRN"],
            "IND": ["PAK", "CHN", "NPL", "BTN", "BGD", "MMR", "LKA"],
            "NPL": ["CHN", "IND"], "BTN": ["CHN", "IND"],
            "BGD": ["IND", "MMR"], "MMR": ["BGD", "IND", "CHN", "LAO", "THA"],
            "LAO": ["MMR", "CHN", "VNM", "KHM", "THA"],
            "THA": ["MMR", "LAO", "KHM", "MYS"],
            "KHM": ["THA", "LAO", "VNM"],
            "VNM": ["CHN", "LAO", "KHM"],
            "MYS": ["THA", "SGP", "IDN", "BRN"],
            "SGP": ["MYS"], "IDN": ["MYS", "PNG", "TLS"],
            "PNG": ["IDN"], "TLS": ["IDN"],
            "PHL": [], "BRN": ["MYS"],
            "AUS": [], "NZL": [],
            "FJI": [], "PNG": ["IDN"],
            "SLB": [], "VUT": [], "NCL": [],
            "WSM": [], "TON": [], "KIR": [], "TUV": [], "NRU": [], "MHL": [], "FSM": [], "PLW": [],
            "ZAF": ["NAM", "BWA", "ZWE", "MOZ", "SWZ", "LSO"],
            "NAM": ["ZAF", "BWA", "ZMB", "AGO"],
            "BWA": ["NAM", "ZAF", "ZWE", "ZMB"],
            "ZWE": ["BWA", "ZAF", "MOZ", "ZMB"],
            "MOZ": ["ZWE", "ZAF", "SWZ", "MWI", "ZMB", "TZA"],
            "ZMB": ["NAM", "BWA", "ZWE", "MOZ", "MWI", "COD", "TZA", "AGO"],
            "MWI": ["ZMB", "MOZ", "TZA"],
            "AGO": ["NAM", "ZMB", "COG", "COD"],
            "SWZ": ["ZAF", "MOZ"], "LSO": ["ZAF"],
            "COG": ["GAB", "CMR", "CAF", "COD", "CAB", "AGO"],
            "CAB": ["COG", "COD"],
            "GAB": ["COG", "CMR", "GNQ"],
            "GNQ": ["GAB", "CMR"],
            "CMR": ["NGA", "TCD", "CAF", "COG", "GAB", "GNQ"],
            "CAF": ["CMR", "TCD", "SSD", "COD", "COG"],
            "TCD": ["CAF", "SDN", "NER", "CMR", "LBY"],
            "SDN": ["TCD", "LBY", "EGY", "ERI", "ETH", "SSD", "CAF"],
            "SSD": ["SDN", "CAF", "COD", "UGA", "KEN", "ETH"],
            "ETH": ["ERI", "DJI", "SOM", "KEN", "SSD", "SDN"],
            "ERI": ["SDN", "ETH", "DJI"],
            "DJI": ["ERI", "ETH", "SOM"],
            "SOM": ["DJI", "ETH", "KEN"],
            "KEN": ["SOM", "ETH", "SSD", "UGA", "TZA"],
            "UGA": ["KEN", "TZA", "RWA", "COD", "SSD"],
            "RWA": ["UGA", "TZA", "BDI", "COD"],
            "BDI": ["RWA", "TZA", "COD"],
            "COD": ["UGA", "RWA", "BDI", "TZA", "ZMB", "AGO", "COG", "CAF", "SSD"],
            "TZA": ["KEN", "UGA", "RWA", "BDI", "COD", "ZMB", "MWI", "MOZ"],
            "ZMB": ["TZA", "MWI", "MOZ", "ZWE", "BWA", "NAM", "AGO", "COD"],
            "MWI": ["ZMB", "MOZ", "TZA"],
            "MOZ": ["TZA", "MWI", "ZWE", "ZAF", "SWZ", "ZMB"],
            "ZWE": ["MOZ", "ZMB", "BWA", "ZAF"],
            "BWA": ["ZWE", "ZMB", "NAM", "ZAF"],
            "NAM": ["BWA", "ZMB", "AGO", "ZAF"],
            "AGO": ["NAM", "ZMB", "COG", "COD", "CAB"],
            "ZAF": ["NAM", "BWA", "ZWE", "MOZ", "SWZ", "LSO"],
            "SWZ": ["ZAF", "MOZ"], "LSO": ["ZAF"],
        }
    
    def _compute_distances(self) -> None:
        """Compute great-circle distances between all country pairs."""
        for iso3_a, geom_a in self.geometries.items():
            for iso3_b, geom_b in self.geometries.items():
                if iso3_a >= iso3_b:
                    continue
                dist = self._haversine(
                    geom_a.centroid_lat, geom_a.centroid_lon,
                    geom_b.centroid_lat, geom_b.centroid_lon
                )
                self.distance_matrix[(iso3_a, iso3_b)] = dist
                self.distance_matrix[(iso3_b, iso3_a)] = dist
    
    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in kilometers."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def get_distance(self, iso3_a: str, iso3_b: str) -> float:
        """Get distance between two countries."""
        if iso3_a == iso3_b:
            return 0.0
        key = (iso3_a, iso3_b) if iso3_a < iso3_b else (iso3_b, iso3_a)
        return self.distance_matrix.get(key, 20000.0)  # Default ~half earth circumference
    
    def are_neighbors(self, iso3_a: str, iso3_b: str) -> bool:
        """Check if two countries share a border."""
        geom_a = self.geometries.get(iso3_a)
        if not geom_a:
            return False
        return iso3_b in geom_a.neighbors
    
    def get_neighbors(self, iso3: str) -> List[str]:
        """Get neighboring countries."""
        geom = self.geometries.get(iso3)
        return geom.neighbors if geom else []
    
    def calculate_trade_gravity(
        self,
        exporter: Country,
        importer: Country,
        state: SimulationState,
    ) -> float:
        """Calculate expected trade flow using gravity model."""
        # Basic gravity: GDP_exporter * GDP_importer / distance^beta
        beta = self.config.get("trade_distance_elasticity", 1.5)
        
        gdp_product = exporter.gdp_usd * importer.gdp_usd
        distance = self.get_distance(exporter.iso3, importer.iso3)
        
        if distance == 0:
            distance = 100  # Minimum distance for same country
        
        base_flow = gdp_product / (distance ** beta)
        
        # Adjustments
        # Contiguity bonus
        if self.are_neighbors(exporter.iso3, importer.iso3):
            base_flow *= 2.5
        
        # Trade agreement bonus
        # Would check treaties/alliances here
        
        # Sanctions penalty
        if importer.iso3 in exporter.sanctions_on or exporter.iso3 in importer.sanctions_by:
            base_flow *= 0.1
        
        # Diplomatic relations
        rel = exporter.diplomatic_relations.get(importer.iso3, 0)
        base_flow *= (1 + rel / 200)  # -50% to +50%
        
        # Resource complementarity
        # Exporter has resources importer needs
        resource_bonus = 1.0
        for resource, amount in exporter.resources.items():
            if amount > 0 and importer.resources.get(resource, 0) == 0:
                resource_bonus += 0.2
        
        base_flow *= resource_bonus
        
        # Add noise
        base_flow *= np.random.lognormal(0, 0.5)
        
        return max(0, base_flow)
    
    def calculate_conflict_contagion(
        self,
        conflict_country: str,
        target_country: str,
        state: SimulationState,
    ) -> float:
        """Calculate probability of conflict spreading to neighbor."""
        if not self.are_neighbors(conflict_country, target_country):
            return 0.0
        
        base_prob = 0.05
        
        # Ethnic kinship
        # Would check shared ethnic groups
        
        # Refugee flows
        conflict = None
        for c in state.conflicts.values():
            if c.primary_attacker == conflict_country or c.primary_defender == conflict_country:
                conflict = c
                break
        
        if conflict:
            refugees = conflict.displaced_persons
            if refugees > 100000:
                base_prob += 0.1
            elif refugees > 10000:
                base_prob += 0.05
        
        # Regime similarity (instability contagion)
        conflict_regime = state.countries[conflict_country].regime_type
        target_regime = state.countries[target_country].regime_type
        conflict_regime_str = conflict_regime.value if hasattr(conflict_regime, 'value') else conflict_regime
        target_regime_str = target_regime.value if hasattr(target_regime, 'value') else target_regime
        if conflict_regime_str == target_regime_str and conflict_regime_str in ["anocracy", "failed_state"]:
            base_prob += 0.1
        
        # Alliance obligations
        # Would check alliances
        
        return min(base_prob, 0.5)
    
    def calculate_migration_potential(
        self,
        origin: Country,
        destination: Country,
    ) -> float:
        """Calculate migration potential between countries."""
        # Push factors (origin)
        push = 0.0
        push += (1 - origin.stability_index) * 0.3
        push += max(0, -origin.gdp_growth_rate) * 2
        push += origin.unemployment_rate / 100 * 0.5
        push += origin.conflict_risk if hasattr(origin, 'conflict_risk') else 0
        
        # Pull factors (destination)
        pull = 0.0
        pull += destination.stability_index * 0.3
        pull += max(0, destination.gdp_growth_rate) * 2
        pull += (1 - destination.unemployment_rate / 100) * 0.5
        pull += destination.gdp_per_capita_usd / 50000 * 0.3
        
        # Distance decay
        distance = self.get_distance(origin.iso3, destination.iso3)
        distance_factor = np.exp(-distance / 3000)  # Half-life ~3000km
        
        # Diaspora/network effect
        # Would check existing migrant stock
        network_factor = 1.0
        
        # Policy barriers
        # Would check visa policies, immigration laws
        policy_factor = 1.0
        
        return (push + pull) * distance_factor * network_factor * policy_factor
    
    def update_geometries_from_countries(self, countries: Dict[str, Country]) -> None:
        """Update geometry data from country objects."""
        for iso3, country in countries.items():
            if iso3 in self.geometries:
                self.geometries[iso3].area_km2 = country.area_km2


def build_spatial_graph(state: SimulationState, spatial: SpatialEngine) -> nx.Graph:
    """Build a graph of spatial relationships for analysis."""
    G = nx.Graph()
    
    for iso3, country in state.countries.items():
        regime_val = country.regime_type.value if hasattr(country.regime_type, 'value') else country.regime_type
        G.add_node(iso3, 
                   gdp=country.gdp_usd,
                   population=country.population,
                   regime=regime_val,
                   stability=country.stability_index)
    
    # Add edges for neighbors
    for iso3, country in state.countries.items():
        for neighbor in spatial.get_neighbors(iso3):
            if neighbor in state.countries:
                dist = spatial.get_distance(iso3, neighbor)
                G.add_edge(iso3, neighbor, distance=dist, type="border")
    
    # Add edges for trade relationships
    for iso3, country in state.countries.items():
        for partner_iso3, volume in country.trade_partners.items():
            if partner_iso3 in state.countries and volume > 0:
                if G.has_edge(iso3, partner_iso3):
                    G[iso3][partner_iso3]["trade_volume"] = volume
                else:
                    G.add_edge(iso3, partner_iso3, trade_volume=volume, type="trade")
    
    # Add edges for alliances
    for alliance in state.alliances.values():
        members = alliance.members
        for i, m1 in enumerate(members):
            for m2 in members[i+1:]:
                if m1 in state.countries and m2 in state.countries:
                    if G.has_edge(m1, m2):
                        G[m1][m2]["alliance"] = True
                    else:
                        G.add_edge(m1, m2, alliance=True, type="alliance")
    
    return G