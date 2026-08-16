"""Benchmarks for the geopolitical-data-generator engine."""

import pytest
import pandas as pd
from tests.conftest import calibration_data


@pytest.fixture
def calibration_data():
    """Provide calibration fixture data."""
    return pd.read_parquet("calibration/fixtures/wdi.parquet")


def bench_01_calibration(benchmark, calibration_data):
    """Benchmark Bayesian calibration performance."""
    from calibration.bayesian_calibrator import calibrate

    benchmark(calibrate, "calibration/fixtures", "/tmp/cal_out.json")


def bench_02_dag_execution(benchmark, calibration_data):
    """Benchmark DAG scenario execution performance."""
    from engine.composition import ScenarioDAG, DAGNode

    # Create a simple DAG
    dag = ScenarioDAG()
    node = DAGNode(id="n1", type="scenario")
    dag.add_node(node)

    # Benchmark DAG execution
    benchmark(lambda: dag.execute(None))


def bench_03_agent_update(benchmark):
    """Benchmark agent state update performance."""
    from engine.agents import Household, Firm, AgentType

    # Create agents
    h = Household(agent_id="h1", agent_type=AgentType.HOUSEHOLD)
    f = Firm(agent_id="f1", agent_type=AgentType.FIRM)

    # Benchmark update
    benchmark(
        lambda: (
            h.update({"prices": 1.0, "inequality": 0.1}),
            f.update({"demand": 1.0, "cost": 1.0}),
        )
    )
