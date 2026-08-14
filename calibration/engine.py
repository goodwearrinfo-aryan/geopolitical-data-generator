"""Calibration engine for parameter estimation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize

from schemas.core import SimulationState, Country, RegimeType, Conflict, ConflictType
from engine.temporal import TemporalEngine


@dataclass
class CalibrationTarget:
    """A target for calibration with observed data."""
    name: str
    observed_values: np.ndarray
    predicted_values: np.ndarray
    weight: float = 1.0
    transform: Optional[str] = None  # log, sqrt, none


@dataclass
class CalibrationResult:
    """Results from calibration."""
    parameter_names: List[str]
    optimal_values: np.ndarray
    objective_value: float
    convergence: bool
    iterations: int
    covariance: Optional[np.ndarray] = None
    confidence_intervals: Optional[List[Tuple[float, float]]] = None


class CalibrationEngine:
    """Calibrates model parameters against historical data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibration_config = config.get("calibration", {})
        
        # Data sources
        self.world_bank_data: Dict[str, pd.DataFrame] = {}
        self.inscr_data: Dict[str, pd.DataFrame] = {}
        self.acled_data: Optional[pd.DataFrame] = None
        
        # Parameter bounds
        self.param_bounds: Dict[str, Tuple[float, float]] = {}
        self.param_names: List[str] = []
        
        # Targets
        self.targets: List[CalibrationTarget] = []
    
    def load_world_bank_data(self, data_dir: str) -> None:
        """Load World Bank WDI data."""
        path = Path(data_dir)
        if not path.exists():
            return
        
        # Expected files: GDP, population, trade, etc.
        for csv_file in path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                key = csv_file.stem
                self.world_bank_data[key] = df
            except Exception as e:
                print(f"Failed to load {csv_file}: {e}")
    
    def load_inscr_data(self, data_dir: str) -> None:
        """Load INSCR/CSP data."""
        path = Path(data_dir)
        if not path.exists():
            return
        
        for csv_file in path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                key = csv_file.stem
                self.inscr_data[key] = df
            except Exception as e:
                print(f"Failed to load {csv_file}: {e}")
    
    def load_acled_data(self, data_path: str) -> None:
        """Load ACLED conflict data."""
        try:
            self.acled_data = pd.read_csv(data_path)
        except Exception as e:
            print(f"Failed to load ACLED data: {e}")
    
    def define_parameters(self) -> Dict[str, Tuple[float, float]]:
        """Define parameters to calibrate with bounds."""
        bounds = {
            # Political
            "coup_base_rate": (0.005, 0.05),
            "coup_stability_sensitivity": (1.0, 10.0),
            "protest_threshold": (0.3, 0.8),
            "protest_stability_sensitivity": (1.0, 8.0),
            "regime_transition_prob": (0.001, 0.05),
            "election_legitimacy_boost": (0.01, 0.1),
            
            # Conflict
            "escalation_base_prob": (0.05, 0.3),
            "deescalation_base_prob": (0.05, 0.4),
            "conflict_casualty_rate": (100, 10000),
            "conflict_economic_impact": (0.01, 0.1),
            "nuclear_threshold": (0.001, 0.05),
            
            # Economic
            "gdp_growth_trend": (0.01, 0.05),
            "gdp_shock_std": (0.01, 0.05),
            "trade_distance_elasticity": (1.0, 2.5),
            "sanction_gdp_impact": (0.01, 0.1),
            "resource_growth_boost": (0.005, 0.03),
            
            # Diplomatic
            "alliance_formation_rate": (0.01, 0.1),
            "alliance_dissolution_rate": (0.005, 0.05),
            "sanction_imposition_rate": (0.01, 0.1),
            "treaty_compliance_rate": (0.5, 0.95),
            
            # Demographic
            "migration_push_sensitivity": (0.1, 1.0),
            "urbanization_growth_boost": (0.005, 0.02),
        }
        
        self.param_bounds = bounds
        self.param_names = list(bounds.keys())
        return bounds
    
    def create_targets_from_data(self, state: SimulationState) -> List[CalibrationTarget]:
        """Create calibration targets from loaded data."""
        targets = []
        
        # GDP targets from World Bank
        if "gdp" in self.world_bank_data:
            gdp_df = self.world_bank_data["gdp"]
            # Match countries and years
            for country in state.countries.values():
                country_data = gdp_df[gdp_df["iso3"] == country.iso3]
                if len(country_data) > 0:
                    observed = country_data["gdp_usd"].values
                    predicted = np.array([country.gdp_usd] * len(observed))  # Placeholder
                    targets.append(CalibrationTarget(
                        name=f"gdp_{country.iso3}",
                        observed_values=observed,
                        predicted_values=predicted,
                        weight=1.0,
                        transform="log",
                    ))
        
        # Conflict targets from ACLED
        if self.acled_data is not None:
            for country in state.countries.values():
                country_conflicts = self.acled_data[
                    (self.acled_data["country"] == country.iso3) &
                    (self.acled_data["year"] >= 2010)
                ]
                if len(country_conflicts) > 0:
                    observed = country_conflicts.groupby("year")["fatalities"].sum().values
                    predicted = np.zeros_like(observed)  # Placeholder
                    targets.append(CalibrationTarget(
                        name=f"conflict_{country.iso3}",
                        observed_values=observed,
                        predicted_values=predicted,
                        weight=2.0,  # Higher weight for conflict
                        transform="sqrt",
                    ))
        
        # Regime targets from INSCR
        if "polity" in self.inscr_data:
            polity_df = self.inscr_data["polity"]
            for country in state.countries.values():
                country_polity = polity_df[polity_df["iso3"] == country.iso3]
                if len(country_polity) > 0:
                    observed = country_polity["polity2"].values
                    # Map polity to regime probability
                    predicted = np.array([
                        self._polity_to_regime_prob(country.regime_type, p)
                        for p in observed
                    ])
                    targets.append(CalibrationTarget(
                        name=f"regime_{country.iso3}",
                        observed_values=observed,
                        predicted_values=predicted,
                        weight=1.5,
                    ))
        
        self.targets = targets
        return targets
    
    def _polity_to_regime_prob(self, regime: RegimeType, polity: int) -> float:
        """Convert polity score to regime probability."""
        # Polity: -10 to 10
        # Democracy: 6 to 10, Autocracy: -10 to -6, Anocracy: -5 to 5
        if regime == RegimeType.DEMOCRACY:
            return max(0, (polity - 5) / 5)
        elif regime == RegimeType.AUTOCRACY:
            return max(0, (-5 - polity) / 5)
        else:
            return max(0, 1 - abs(polity) / 5)
    
    def objective_function(self, params: np.ndarray, state: SimulationState) -> float:
        """Compute weighted sum of squared errors."""
        # Update model parameters
        param_dict = dict(zip(self.param_names, params))
        
        # Run simulation with these parameters (simplified)
        # In reality, would run full simulation
        predicted_state = self._run_calibration_simulation(param_dict, state)
        
        # Compute errors
        total_error = 0.0
        for target in self.targets:
            if target.transform == "log":
                obs = np.log(np.maximum(target.observed_values, 1))
                pred = np.log(np.maximum(target.predicted_values, 1))
            elif target.transform == "sqrt":
                obs = np.sqrt(np.maximum(target.observed_values, 0))
                pred = np.sqrt(np.maximum(target.predicted_values, 0))
            else:
                obs = target.observed_values
                pred = target.predicted_values
            
            # Align lengths
            min_len = min(len(obs), len(pred))
            if min_len > 0:
                error = np.mean((obs[:min_len] - pred[:min_len]) ** 2)
                total_error += target.weight * error
        
        return total_error
    
    def _run_calibration_simulation(
        self,
        params: Dict[str, float],
        initial_state: SimulationState,
    ) -> SimulationState:
        """Run a quick simulation for calibration."""
        # This is a placeholder - would run actual simulation
        # For now, return initial state with perturbed values
        import copy
        state = copy.deepcopy(initial_state)
        
        # Apply parameter effects
        for country in state.countries.values():
            country.gdp_growth_rate = params.get("gdp_growth_trend", 0.025)
            country.coup_risk = params.get("coup_base_rate", 0.02)
        
        return state
    
    def calibrate(
        self,
        initial_state: SimulationState,
        method: str = "hybrid",
        max_iterations: int = 100,
    ) -> CalibrationResult:
        """Run calibration optimization."""
        if not self.param_bounds:
            self.define_parameters()
        
        if not self.targets:
            self.create_targets_from_data(initial_state)
        
        # Initial guess (midpoint of bounds)
        x0 = np.array([(b[0] + b[1]) / 2 for b in self.param_bounds.values()])
        bounds = list(self.param_bounds.values())
        
        if method == "bayesian":
            return self._calibrate_bayesian(x0, bounds, initial_state)
        elif method == "moment_matching":
            return self._calibrate_moment_matching(x0, bounds, initial_state)
        else:  # hybrid
            return self._calibrate_hybrid(x0, bounds, initial_state, max_iterations)
    
    def _calibrate_hybrid(
        self,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        initial_state: SimulationState,
        max_iterations: int,
    ) -> CalibrationResult:
        """Hybrid calibration: global search + local refinement."""
        # Phase 1: Differential evolution (global)
        from scipy.optimize import differential_evolution
        
        result_de = differential_evolution(
            lambda x: self.objective_function(x, initial_state),
            bounds,
            maxiter=min(50, max_iterations // 2),
            popsize=15,
            seed=42,
            disp=False,
        )
        
        # Phase 2: Local refinement with L-BFGS-B
        from scipy.optimize import minimize
        
        result_local = minimize(
            lambda x: self.objective_function(x, initial_state),
            result_de.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations // 2, "disp": False},
        )
        
        # Estimate covariance (inverse Hessian approximation)
        try:
            from scipy.optimize import approx_fprime
            eps = 1e-6
            hess = np.zeros((len(x0), len(x0)))
            for i in range(len(x0)):
                for j in range(len(x0)):
                    # Finite difference Hessian
                    x_plus = result_local.x.copy()
                    x_minus = result_local.x.copy()
                    x_plus[i] += eps
                    x_plus[j] += eps
                    x_minus[i] -= eps
                    x_minus[j] -= eps
                    f_pp = self.objective_function(x_plus, initial_state)
                    f_pm = self.objective_function(x_plus - np.eye(len(x0))[j] * 2 * eps, initial_state)
                    f_mp = self.objective_function(x_minus + np.eye(len(x0))[j] * 2 * eps, initial_state)
                    f_mm = self.objective_function(x_minus, initial_state)
                    hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
            
            cov = np.linalg.inv(hess + np.eye(len(x0)) * 1e-6)
            ci = []
            for i in range(len(x0)):
                se = np.sqrt(cov[i, i])
                ci.append((result_local.x[i] - 1.96 * se, result_local.x[i] + 1.96 * se))
        except:
            cov = None
            ci = None
        
        return CalibrationResult(
            parameter_names=self.param_names,
            optimal_values=result_local.x,
            objective_value=result_local.fun,
            convergence=result_local.success,
            iterations=result_local.nit if hasattr(result_local, 'nit') else 0,
            covariance=cov,
            confidence_intervals=ci,
        )
    
    def _calibrate_moment_matching(
        self,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        initial_state: SimulationState,
    ) -> CalibrationResult:
        """Moment matching calibration (faster, approximate)."""
        # Match key moments: mean GDP growth, conflict frequency, regime stability
        target_moments = self._compute_target_moments()
        
        def moment_error(params):
            param_dict = dict(zip(self.param_names, params))
            sim_moments = self._simulate_moments(param_dict, initial_state)
            return np.sum((np.array(target_moments) - np.array(sim_moments)) ** 2)
        
        from scipy.optimize import minimize
        result = minimize(
            moment_error,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "disp": False},
        )
        
        return CalibrationResult(
            parameter_names=self.param_names,
            optimal_values=result.x,
            objective_value=result.fun,
            convergence=result.success,
            iterations=result.nit if hasattr(result, 'nit') else 0,
        )
    
    def _calibrate_bayesian(
        self,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        initial_state: SimulationState,
    ) -> CalibrationResult:
        """Bayesian calibration using MCMC (requires pymc)."""
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            print("pymc not available, falling back to hybrid")
            return self._calibrate_hybrid(x0, bounds, initial_state, 100)
        
        # This is a simplified version - full implementation would be more complex
        with pm.Model() as model:
            # Priors
            params = {}
            for i, name in enumerate(self.param_names):
                low, high = bounds[i]
                params[name] = pm.Uniform(name, lower=low, upper=high)
            
            # Likelihood (simplified)
            # Would need to define proper likelihood based on targets
            
            # Sample
            trace = pm.sample(
                draws=self.calibration_config.get("mcmc_draws", 1000),
                tune=self.calibration_config.get("mcmc_tune", 500),
                chains=self.calibration_config.get("mcmc_chains", 4),
                cores=1,
                progressbar=False,
            )
        
        # Extract posterior means
        posterior_means = [trace.posterior[name].mean().values.item() for name in self.param_names]
        
        return CalibrationResult(
            parameter_names=self.param_names,
            optimal_values=np.array(posterior_means),
            objective_value=0.0,  # Not directly available
            convergence=True,
            iterations=len(trace.posterior.draw),
            confidence_intervals=[
                (trace.posterior[name].quantile(0.025).values.item(),
                 trace.posterior[name].quantile(0.975).values.item())
                for name in self.param_names
            ],
        )
    
    def _compute_target_moments(self) -> List[float]:
        """Compute target moments from data."""
        moments = []
        
        # Global GDP growth
        if "gdp" in self.world_bank_data:
            gdp_df = self.world_bank_data["gdp"]
            global_growth = gdp_df.groupby("year")["gdp_usd"].sum().pct_change().mean()
            moments.append(global_growth if not np.isnan(global_growth) else 0.025)
        else:
            moments.append(0.025)
        
        # Conflict frequency
        if self.acled_data is not None:
            conflict_freq = self.acled_data.groupby("year")["event_id"].nunique().mean()
            moments.append(conflict_freq if not np.isnan(conflict_freq) else 100)
        else:
            moments.append(100)
        
        # Regime stability (Polity persistence)
        if "polity" in self.inscr_data:
            polity_df = self.inscr_data["polity"]
            # Compute year-over-year correlation
            polity_pivot = polity_df.pivot(index="country", columns="year", values="polity2")
            if polity_pivot.shape[1] > 1:
                autocorr = polity_pivot.corrwith(polarity_pivot.shift(1, axis=1)).mean()
                moments.append(autocorr if not np.isnan(autocorr) else 0.95)
            else:
                moments.append(0.95)
        else:
            moments.append(0.95)
        
        return moments
    
    def _simulate_moments(
        self,
        params: Dict[str, float],
        initial_state: SimulationState,
    ) -> List[float]:
        """Simulate moments for given parameters."""
        # Placeholder - would run simulation and compute moments
        return [
            params.get("gdp_growth_trend", 0.025),
            100 * params.get("escalation_base_prob", 0.15),
            0.95 - params.get("regime_transition_prob", 0.01) * 10,
        ]
    
    def validate(
        self,
        calibrated_params: np.ndarray,
        initial_state: SimulationState,
        validation_years: int = 10,
    ) -> Dict[str, float]:
        """Validate calibrated model against out-of-sample data."""
        # Run simulation with calibrated parameters
        param_dict = dict(zip(self.param_names, calibrated_params))
        final_state = self._run_calibration_simulation(param_dict, initial_state)
        
        metrics = {}
        
        # GDP RMSE
        if "gdp" in self.world_bank_data:
            gdp_df = self.world_bank_data["gdp"]
            # Compare last validation_years
            errors = []
            for country in final_state.countries.values():
                country_data = gdp_df[
                    (gdp_df["iso3"] == country.iso3) &
                    (gdp_df["year"] >= gdp_df["year"].max() - validation_years)
                ]
                if len(country_data) > 0:
                    obs = country_data["gdp_usd"].values
                    pred = np.full_like(obs, country.gdp_usd)
                    if len(obs) == len(pred):
                        errors.append(np.sqrt(np.mean((obs - pred) ** 2)))
            
            if errors:
                metrics["gdp_rmse"] = float(np.mean(errors))
        
        # Conflict AUC
        if self.acled_data is not None:
            # Binary classification: conflict > threshold
            auc_scores = []
            for country in final_state.countries.values():
                country_conflicts = self.acled_data[
                    (self.acled_data["country"] == country.iso3) &
                    (self.acled_data["year"] >= self.acled_data["year"].max() - validation_years)
                ]
                if len(country_conflicts) > 0:
                    y_true = (country_conflicts.groupby("year")["fatalities"].sum() > 100).astype(int).values
                    y_pred = np.full_like(y_true, 0.5)  # Placeholder
                    if len(np.unique(y_true)) > 1:
                        from sklearn.metrics import roc_auc_score
                        try:
                            auc = roc_auc_score(y_true, y_pred)
                            auc_scores.append(auc)
                        except:
                            pass
            
            if auc_scores:
                metrics["conflict_auc"] = float(np.mean(auc_scores))
        
        # Regime transition accuracy
        if "polity" in self.inscr_data:
            polity_df = self.inscr_data["polity"]
            correct = 0
            total = 0
            for country in final_state.countries.values():
                country_polity = polity_df[
                    (polity_df["iso3"] == country.iso3) &
                    (polity_df["year"] >= polity_df["year"].max() - validation_years)
                ]
                if len(country_polity) > 1:
                    for _, row in country_polity.iterrows():
                        true_regime = self._polity_to_regime(row["polity2"])
                        pred_regime = country.regime_type
                        if true_regime == pred_regime:
                            correct += 1
                        total += 1
            
            if total > 0:
                metrics["regime_transition_acc"] = correct / total
        
        return metrics
    
    def _polity_to_regime(self, polity: int) -> RegimeType:
        """Convert polity score to regime type."""
        if polity >= 6:
            return RegimeType.DEMOCRACY
        elif polity <= -6:
            return RegimeType.AUTOCRACY
        else:
            return RegimeType.ANOCRACY