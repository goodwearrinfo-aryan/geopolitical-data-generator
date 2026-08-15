"""Ensemble engine for Monte Carlo simulation and sensitivity analysis."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import pandas as pd

from schemas.core import SimulationState, Country, RegimeType
from engine.temporal import TemporalEngine, TimestepFrequency
from engine.causal import CausalEngine, ConflictDynamics
from engine.spatial import SpatialEngine


@dataclass
class EnsembleResult:
    """Results from a single ensemble run."""
    run_id: int
    seed: int
    final_state: SimulationState
    metrics: Dict[str, float]
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summary statistics
    global_gdp_trajectory: List[float] = field(default_factory=list)
    global_conflict_trajectory: List[int] = field(default_factory=list)
    regime_changes: int = 0
    conflicts_started: int = 0
    wars: int = 0
    nuclear_events: int = 0


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""
    parameter: str
    method: str  # morris, sobol
    mu_star: float  # Morris mean elementary effect
    sigma: float    # Morris std of elementary effects
    sobol_first: Optional[float] = None  # First-order Sobol index
    sobol_total: Optional[float] = None  # Total-order Sobol index
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


class EnsembleEngine:
    """Runs ensemble simulations and sensitivity analysis."""
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        n_workers: int = 4,
        seed: int = 42,
    ):
        # Store only picklable config data
        self.base_config = self._make_picklable(base_config)
        self.n_workers = n_workers
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Results storage
        self.results: List[EnsembleResult] = []
        self.sensitivity_results: List[SensitivityResult] = []
    
    def _make_picklable(self, obj: Any) -> Any:
        """Convert config to picklable format (dict with primitive types)."""
        if hasattr(obj, '__dict__'):
            # Dataclass or object with __dict__
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith("_"):
                    result[key] = self._make_picklable(value)
            return result
        elif isinstance(obj, dict):
            return {k: self._make_picklable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_picklable(v) for v in obj]
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        else:
            return obj
    
    def run_ensemble(
        self,
        n_scenarios: int,
        parameter_variations: Optional[Dict[str, Tuple[float, float]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EnsembleResult]:
        """Run ensemble of simulations with parameter variations."""
        
        if parameter_variations is None:
            parameter_variations = self._get_default_variations()
        
        # Generate seeds for each run
        seeds = [self.rng.integers(0, 2**32) for _ in range(n_scenarios)]
        
        # Generate parameter sets
        param_sets = self._generate_parameter_sets(parameter_variations, n_scenarios)
        
        # Run in parallel
        self.results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_scenario,
                    run_id,
                    seeds[run_id],
                    param_sets[run_id],
                ): run_id
                for run_id in range(n_scenarios)
            }
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                self.results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, n_scenarios)
        
        # Sort by run_id
        self.results.sort(key=lambda r: r.run_id)
        return self.results
    
    def _run_single_scenario(
        self,
        run_id: int,
        seed: int,
        parameters: Dict[str, float],
    ) -> EnsembleResult:
        """Run a single simulation scenario."""
        # This would normally instantiate the full simulation
        # For now, return a placeholder with simulated metrics
        rng = np.random.default_rng(seed)
        
        # Simulate trajectory
        n_steps = 360  # 30 years monthly
        trajectory = []
        
        global_gdp = 100_000_000_000_000  # $100T
        active_conflicts = 10
        
        for step in range(n_steps):
            # GDP growth with shocks
            growth = rng.normal(0.025, 0.015)  # 2.5% ± 1.5%
            for param, value in parameters.items():
                if "gdp" in param.lower():
                    growth += value * 0.01
            global_gdp *= (1 + growth / 12)  # Monthly
            
            # Conflict dynamics
            conflict_change = rng.normal(0, 0.5)
            active_conflicts = max(0, int(active_conflicts + conflict_change))
            
            trajectory.append({
                "timestep": step,
                "global_gdp": global_gdp,
                "active_conflicts": active_conflicts,
            })
        
        # Final metrics
        metrics = {
            "final_global_gdp": global_gdp,
            "avg_growth_rate": (global_gdp / 100_000_000_000_000) ** (1/30) - 1,
            "peak_conflicts": max(t["active_conflicts"] for t in trajectory),
            "final_conflicts": active_conflicts,
            "gdp_volatility": np.std([t["global_gdp"] for t in trajectory]),
            "conflict_volatility": np.std([t["active_conflicts"] for t in trajectory]),
        }
        
        return EnsembleResult(
            run_id=run_id,
            seed=seed,
            final_state=None,  # Would be actual state
            metrics=metrics,
            trajectory=trajectory,
            global_gdp_trajectory=[t["global_gdp"] for t in trajectory],
            global_conflict_trajectory=[t["active_conflicts"] for t in trajectory],
        )
    
    def _get_default_variations(self) -> Dict[str, Tuple[float, float]]:
        """Get default parameter variations for ensemble."""
        return {
            "domains.political.coup_base_rate": (0.01, 0.05),
            "domains.conflict.escalation_probability": (0.05, 0.3),
            "domains.economic.gdp_shock_std": (0.01, 0.05),
            "domains.diplomatic.alliance_formation_rate": (0.01, 0.1),
            "calibration.priors.regime_transition": (0.5, 2.0),
        }
    
    def _generate_parameter_sets(
        self,
        variations: Dict[str, Tuple[float, float]],
        n_sets: int,
    ) -> List[Dict[str, float]]:
        """Generate parameter sets using Latin Hypercube Sampling."""
        param_names = list(variations.keys())
        bounds = np.array([variations[p] for p in param_names])
        
        # Latin Hypercube Sampling
        n_params = len(param_names)
        samples = np.zeros((n_sets, n_params))
        
        for i in range(n_params):
            # Divide range into n_sets strata
            strata = np.linspace(0, 1, n_sets + 1)
            # Sample one point per stratum
            points = self.rng.uniform(strata[:-1], strata[1:], n_sets)
            # Shuffle
            self.rng.shuffle(points)
            # Scale to bounds
            samples[:, i] = bounds[i, 0] + points * (bounds[i, 1] - bounds[i, 0])
        
        return [
            {param_names[j]: samples[i, j] for j in range(n_params)}
            for i in range(n_sets)
        ]
    
    def run_morris_sensitivity(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        n_trajectories: int = 20,
        n_steps: int = 4,
    ) -> List[SensitivityResult]:
        """Run Morris elementary effects sensitivity analysis."""
        param_names = list(parameter_bounds.keys())
        n_params = len(param_names)
        
        # Delta for finite differences
        delta = 1.0 / (n_steps - 1)
        
        elementary_effects = {name: [] for name in param_names}
        
        for traj in range(n_trajectories):
            # Generate base point
            base = {}
            for name, (low, high) in parameter_bounds.items():
                base[name] = self.rng.uniform(low, high)
            
            # Generate trajectory
            current = base.copy()
            
            for step in range(n_params):
                # Choose parameter to perturb
                param_idx = (traj * n_params + step) % n_params
                param_name = param_names[param_idx]
                
                # Perturb
                low, high = parameter_bounds[param_name]
                step_size = (high - low) * delta
                perturbed = current.copy()
                perturbed[param_name] = min(high, current[param_name] + step_size)
                
                # Evaluate model at both points
                y_base = self._evaluate_model(current)
                y_perturbed = self._evaluate_model(perturbed)
                
                # Calculate elementary effect
                ee = (y_perturbed - y_base) / step_size * (high - low)
                elementary_effects[param_name].append(ee)
                
                current = perturbed
        
        # Compute statistics
        results = []
        for name in param_names:
            ees = np.array(elementary_effects[name])
            mu_star = np.mean(np.abs(ees))
            sigma = np.std(ees)
            
            # Confidence interval via bootstrap
            ci = self._bootstrap_ci(ees)
            
            results.append(SensitivityResult(
                parameter=name,
                method="morris",
                mu_star=float(mu_star),
                sigma=float(sigma),
                confidence_interval=ci,
            ))
        
        self.sensitivity_results.extend(results)
        return results
    
    def run_sobol_sensitivity(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        n_samples: int = 1000,
    ) -> List[SensitivityResult]:
        """Run Sobol sensitivity analysis (first and total order indices)."""
        param_names = list(parameter_bounds.keys())
        n_params = len(param_names)
        
        # Generate Saltelli samples
        # A and B matrices
        A = self._generate_saltelli_matrix(n_samples, n_params, parameter_bounds)
        B = self._generate_saltelli_matrix(n_samples, n_params, parameter_bounds)
        
        # Evaluate model
        y_A = np.array([self._evaluate_model(dict(zip(param_names, row))) for row in A])
        y_B = np.array([self._evaluate_model(dict(zip(param_names, row))) for row in B])
        
        # Compute indices
        results = []
        var_y = np.var(np.concatenate([y_A, y_B]))
        
        for i, name in enumerate(param_names):
            # Create C matrix (A with column i from B)
            C = A.copy()
            C[:, i] = B[:, i]
            y_C = np.array([self._evaluate_model(dict(zip(param_names, row))) for row in C])
            
            # First-order index
            S_i = np.mean(y_B * (y_C - y_A)) / var_y
            
            # Total-order index
            # Create D matrix (B with column i from A)
            D = B.copy()
            D[:, i] = A[:, i]
            y_D = np.array([self._evaluate_model(dict(zip(param_names, row))) for row in D])
            ST_i = 1 - np.mean(y_A * (y_D - y_B)) / var_y
            
            results.append(SensitivityResult(
                parameter=name,
                method="sobol",
                mu_star=0.0,
                sigma=0.0,
                sobol_first=float(max(0, S_i)),
                sobol_total=float(min(1, max(0, ST_i))),
                confidence_interval=(0.0, 0.0),  # Would bootstrap
            ))
        
        self.sensitivity_results.extend(results)
        return results
    
    def _generate_saltelli_matrix(
        self,
        n_samples: int,
        n_params: int,
        bounds: Dict[str, Tuple[float, float]],
    ) -> np.ndarray:
        """Generate Saltelli sampling matrix."""
        # Use Sobol sequence for quasi-random sampling
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=self.rng.integers(0, 2**32))
        sample = sampler.random(n_samples)
        
        # Scale to bounds
        param_names = list(bounds.keys())
        scaled = np.zeros_like(sample)
        for i, name in enumerate(param_names):
            low, high = bounds[name]
            scaled[:, i] = low + sample[:, i] * (high - low)
        
        return scaled
    
    def _evaluate_model(self, parameters: Dict[str, float]) -> float:
        """Evaluate model output for given parameters.
        
        This is a placeholder - in reality this would run a full simulation
        and return a scalar metric (e.g., final global GDP, conflict count, etc.)
        """
        # Simple proxy metric
        output = 0.0
        for param, value in parameters.items():
            if "coup" in param:
                output -= value * 100
            elif "gdp" in param:
                output -= value * 50
            elif "escalation" in param:
                output -= value * 200
            elif "alliance" in param:
                output += value * 50
            elif "regime" in param:
                output -= value * 30
        
        # Add noise
        output += self.rng.normal(0, 10)
        
        return output
    
    def _bootstrap_ci(self, data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        if len(data) < 2:
            return (0.0, 0.0)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = self.rng.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        return (float(lower), float(upper))
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across ensemble runs."""
        if not self.results:
            return {}
        
        # Collect metrics
        metrics_keys = self.results[0].metrics.keys()
        summary = {}
        
        for key in metrics_keys:
            values = [r.metrics[key] for r in self.results]
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }
        
        # Trajectory statistics
        if self.results[0].global_gdp_trajectory:
            n_steps = len(self.results[0].global_gdp_trajectory)
            gdp_trajectories = np.array([r.global_gdp_trajectory for r in self.results])
            conflict_trajectories = np.array([r.global_conflict_trajectory for r in self.results])
            
            summary["gdp_trajectory"] = {
                "mean": gdp_trajectories.mean(axis=0).tolist(),
                "std": gdp_trajectories.std(axis=0).tolist(),
                "q25": np.percentile(gdp_trajectories, 25, axis=0).tolist(),
                "q75": np.percentile(gdp_trajectories, 75, axis=0).tolist(),
            }
            
            summary["conflict_trajectory"] = {
                "mean": conflict_trajectories.mean(axis=0).tolist(),
                "std": conflict_trajectories.std(axis=0).tolist(),
                "q25": np.percentile(conflict_trajectories, 25, axis=0).tolist(),
                "q75": np.percentile(conflict_trajectories, 75, axis=0).tolist(),
            }
        
        return summary
    
    def export_results(self, output_dir: str) -> None:
        """Export ensemble results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Summary statistics
        summary = self.get_summary_statistics()
        import json
        with open(os.path.join(output_dir, "ensemble_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Individual run metrics
        metrics_df = pd.DataFrame([
            {"run_id": r.run_id, "seed": r.seed, **r.metrics}
            for r in self.results
        ])
        metrics_df.to_parquet(os.path.join(output_dir, "ensemble_metrics.parquet"))
        metrics_df.to_csv(os.path.join(output_dir, "ensemble_metrics.csv"), index=False)
        
        # Sensitivity results
        if self.sensitivity_results:
            sens_df = pd.DataFrame([
                {
                    "parameter": r.parameter,
                    "method": r.method,
                    "mu_star": r.mu_star,
                    "sigma": r.sigma,
                    "sobol_first": r.sobol_first,
                    "sobol_total": r.sobol_total,
                    "ci_lower": r.confidence_interval[0],
                    "ci_upper": r.confidence_interval[1],
                }
                for r in self.sensitivity_results
            ])
            sens_df.to_parquet(os.path.join(output_dir, "sensitivity_results.parquet"))
            sens_df.to_csv(os.path.join(output_dir, "sensitivity_results.csv"), index=False)
        
        print(f"Exported ensemble results to {output_dir}")


def create_ensemble_engine(config: Any) -> EnsembleEngine:
    """Factory function to create ensemble engine from config."""
    # Handle both dict and dataclass config
    if hasattr(config, '__dict__'):
        # Dataclass
        ensemble_config = config.ensemble
        n_workers = getattr(ensemble_config, 'n_workers', 4)
        seed = config.simulation.seed
    else:
        # Dict
        ensemble_config = config.get("ensemble", {})
        n_workers = ensemble_config.get("n_workers", 4)
        seed = config.get("simulation", {}).get("seed", 42)
    
    return EnsembleEngine(
        base_config=config,
        n_workers=n_workers,
        seed=seed,
    )