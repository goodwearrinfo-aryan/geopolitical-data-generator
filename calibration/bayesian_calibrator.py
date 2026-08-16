"""Bayesian calibration module for geopolitical scenario parameters.

Uses PyMC for Hamiltonian Monte Carlo (HMC) / No-U-Turn Sampling (NUTS)
to calibrate model parameters against historical calibration data.
"""

from __future__ import annotations

import json
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path


class BayesianCalibrator:
    """Bayesian calibrator using PyMC for HMC/NUTS sampling.

    Matches calibration data (historical observables) to model parameters
    by computing the posterior distribution via MCMC.
    """

    def __init__(self, fixture_data: pd.DataFrame):
        """Initialize the calibrator with calibration fixture data.

        Args:
            fixture_data: DataFrame with columns:
                - coup_occurred: float (binary, 0/1)
                - gdp_growth_rate: float
                - escalation_events: int (count >= 0)
                - new_alliances: int (count >= 0)
        """
        self.data = fixture_data.copy()
        self.trace = None
        self.posterior_summary = None

    def _build_likelihoods(self, model):
        """Add likelihood terms to the PyMC model.

        All priors are weakly informative; likelihoods link to the calibration
        fixture observables.
        """
        # 1. Coup occurrences (Bernoulli)
        # Prior: Beta(2, 50) - small prior probability of coups
        coup_base_rate = pm.Beta("coup_base_rate", alpha=2.0, beta=50.0)  # noqa: F841
        obs_coups = self.data["coup_occurred"].values
        pm.Bernoulli("coup_likelihood", p=coup_base_rate, observed=obs_coups)

        # 2. GDP growth rate (Normal)
        # Prior: gdp_shock_std ~ HalfNormal(sigma=0.02)
        gdp_shock_std = pm.HalfNormal("gdp_shock_std", sigma=0.02)  # noqa: F841
        obs_gdp_growth = self.data["gdp_growth_rate"].values
        # Placeholder mu; in full implementation links to ScenarioEngine output
        pm.Normal(
            "gdp_likelihood",
            mu=pm.math.constant(0.0),
            sigma=gdp_shock_std,
            observed=obs_gdp_growth,
        )

        # 3. Escalation events (Poisson - handles 0s better than Geometric)
        # Prior: Poisson lambda ~ Gamma(2, 0.1) via exponential
        escalation_lambda = pm.Exponential("escalation_lambda", lam=0.1)  # noqa: F841
        obs_escalations = self.data["escalation_events"].values
        pm.Poisson(
            "escalation_likelihood", mu=escalation_lambda, observed=obs_escalations
        )

        # 4. Alliance formation (Poisson)
        # Prior: alliance_formation ~ Beta(2, 20)
        alliance_formation = pm.Beta("alliance_formation", alpha=2.0, beta=20.0)  # noqa: F841
        obs_alliances = self.data["new_alliances"].values
        pm.Poisson(
            "alliance_likelihood", mu=alliance_formation * 10, observed=obs_alliances
        )

    def build_model(self):
        """Build the PyMC model with priors and likelihoods.

        Returns:
            pm.Model: A PyMC model instance with priors and likelihoods.
        """
        # Use a PyMC model context manager to ensure proper variable registration
        with pm.Model() as model:
            # --- Priors ---
            coup_base_rate = pm.Beta("coup_base_rate", alpha=2.0, beta=50.0)  # noqa: F841
            escalation_lambda = pm.Exponential("escalation_lambda", lam=0.1)  # noqa: F841
            gdp_shock_std = pm.HalfNormal("gdp_shock_std", sigma=0.02)  # noqa: F841
            alliance_formation = pm.Beta("alliance_formation", alpha=2.0, beta=20.0)  # noqa: F841

            # --- Likelihoods ---
            self._build_likelihoods(model)

        return model

    def sample(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
    ) -> az.InferenceData:
        """Run MCMC sampling to obtain the posterior distribution.

        Args:
            draws: Number of posterior draws after tuning.
            tune: Number of tuning (warm-up) samples.
            chains: Number of independent chains.
            target_accept: Target acceptance rate for NUTS sampler.

        Returns:
            az.InferenceData: PyMC trace object containing the posterior samples.
        """
        # Use a model context manager to ensure proper variable registration
        with pm.Model() as model:
            self._build_likelihoods(model)
            # Now sample
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
            )
        self.posterior_summary = az.summary(self.trace, ci_prob=0.9).round(4)
        return self.trace

    def get_posterior_summary(self) -> dict:
        """Return a dict of posterior means, HDIs, and standard errors.

        Returns:
            dict: Summary statistics for each calibrated parameter.
            Format: {"param_name": {"mean": float, "sd": float, "eti90_lb": float, "eti90_ub": float}}
        """
        if self.posterior_summary is None:
            raise ValueError("No posterior summary available. Run .sample() first.")
        # posterior_summary is a SummaryDataFrame with variables as rows
        # and statistics (mean, sd, etc.) as columns
        summary_df = self.posterior_summary
        result = {}
        for param in summary_df.index:
            param_data = {}
            for stat in summary_df.columns:
                param_data[stat] = float(summary_df.loc[param, stat])
            result[param] = param_data
        return result

    def save_calibrated_config(self, output_path: Path | str) -> None:
        """Save the calibrated parameters to a JSON config file.

        The config is formatted for direct use by ScenarioEngine or other
        downstream systems.

        Args:
            output_path: Path to write the JSON config file.
        """
        summary = self.get_posterior_summary()
        config = {
            "coup_base_rate": float(summary["coup_base_rate"]["mean"]),
            "escalation_probability": float(summary["escalation_lambda"]["mean"]),
            "gdp_shock_std": float(summary["gdp_shock_std"]["mean"]),
            "alliance_formation_rate": float(summary["alliance_formation"]["mean"]),
            "posterior_samples": {
                k: [float(v) for v in vs.values.flatten()]
                for k, vs in self.trace.posterior.items()
            },
        }
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def _load_fixtures(fixture_dir: Path | str) -> pd.DataFrame:
        """Load calibration fixture parquet files from a directory.

        Args:
            fixture_dir: Path to directory containing .parquet fixture files.

        Returns:
            pd.DataFrame: Concatenated fixture data with required columns.
        """
        fixtures: list[pd.DataFrame] = []
        p = Path(fixture_dir)
        for fp in sorted(p.glob("*.parquet")):
            df = pd.read_parquet(fp)
            required = {
                "coup_occurred",
                "gdp_growth_rate",
                "escalation_events",
                "new_alliances",
            }
            if not required.issubset(df.columns):
                missing = required - set(df.columns)
                raise ValueError(f"Fixture {fp.name} missing columns: {missing}")
            fixtures.append(df)
        if not fixtures:
            raise FileNotFoundError(f"No parquet fixtures found in {fixture_dir}")
        return pd.concat(fixtures, ignore_index=True)


def calibrate(data_dir: Path | str, output_path: Path | str) -> dict:
    """Convenience function: load fixtures, calibrate, save config.

    Args:
        data_dir: Path to directory containing calibration fixture parquet files.
        output_path: Path to write the calibrated config JSON file.

    Returns:
        dict: Posterior summary statistics.
    """
    fixture_path = Path(data_dir)
    config_path = Path(output_path)

    fixtures = BayesianCalibrator._load_fixtures(fixture_path)
    calibrator = BayesianCalibrator(fixtures)
    calibrator.sample()
    calibrator.save_calibrated_config(config_path)
    return calibrator.get_posterior_summary()


# ------------------------------------------------------------------ #
# Standalone CLI                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python -m calibration.bayesian_calibrator <fixture_dir> <output_json>"
        )
        sys.exit(1)

    fixture_dir = sys.argv[1]
    output_json = sys.argv[2]

    summary = calibrate(fixture_dir, output_json)
    print(f"Calibration complete. Posterior summary:\n{json.dumps(summary, indent=2)}")
