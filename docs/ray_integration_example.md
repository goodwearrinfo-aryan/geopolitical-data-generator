# Ray Parallel Execution Example
# Use Ray for distributed scenario execution and calibration

import ray
ray.init(ignore_reinit_error=True, num_cpus=4)  # Use 4 CPUs

from engine.counterfactuals import DSLRParser, run_counterfactual_analysis
from calibration.bayesian_calibrator import calibrate
import requests

# Parallel counterfactual analysis
interventions = [DSLRParser.parse('do(coup_base_rate=0.1)')[0]]
results = ray.get([
    remote_run_counterfactual.remote(intervention) 
    for intervention in interventions
])

# Parallel API calls for scenario status
@ray.remote
def check_scenario_status(scenario_id):
    r = requests.get(f'http://localhost:8000/api/v1/jobs/')
    return r.json()

statuses = ray.get([check_scenario_status.remote(s) for s in ['ict', 'coup', 'economic', 'conflict']])

# Parallel calibration
ray.get([remote_calibrate.remote() for _ in range(2)])

ray.shutdown()