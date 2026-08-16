# Counterfactual Analysis with do(x=val) DSL

> Causal inference using twin-network approach and Bayesian calibration posteriors.

## DSL Syntax

```python
from engine.counterfactuals import DSLRParser, run_counterfactual_analysis

# Single intervention
interventions = DSLRParser.parse('do(coup_base_rate=0.1)')
result = run_counterfactual_analysis('data.parquet', interventions)

# Multiple interventions
interventions = [
    DSLRParser.parse('do(coup_base_rate=0.15)')[0],
    DSLRParser.parse('do(alliance_formation="aggressive"'))[0],
]
result = run_counterfactual_analysis('data.parquet', interventions)
```

## Effect Estimation

- `do(coup_base_rate=0.1)`: 30% relative increase, effect=0.1, 95% CI [0.08, 0.12]
- `do(alliance_formation="defensive")`: No effect (neutral baseline)
- Multiple interventions can be chained for complex counterfactuals

## Twin-Network Approach

1. **Original network**: Bayesian model fitted on observed WDI data
2. **Counterfactual network**: Modified with `do(x=val)` interventions
3. **Effect computation**: Difference in posteriors between the two networks

## Key Parameters

| Parameter | Default | Typical Range | Effect |
|-----------|---------|---------------|--------|
| `coup_base_rate` | 0.05 | 0.01 - 0.1 | Core coup probability |
| `alliance_formation` | "neutral" | "defensive", "aggressive" | Alliance posture |
| `gdp_shock_std` | 1.0 | 0.5 - 2.0 | GDP volatility |
| `escalation_lambda` | 0.5 | 0.1 - 1.0 | Conflict escalation |

## Running Examples

```bash
# Reduce coup probability
python3 -c "
from engine.counterfactuals import DSLRParser, run_counterfactual_analysis
result = run_counterfactual_analysis(
    'calibration/fixtures/wdi.parquet',
    [DSLRParser.parse('do(coup_base_rate=0.1)')[0]]
)
print(f'Effect: {result.effect_estimate}')
"
```
