# AutoEmulate 2.0.0 API Migration Guide

This document outlines the key differences between the previous version of AutoEmulate and the stable 2.0.0 release. It provides a guide for migrating your code and examples of common workflows.

## Key Differences

| Feature | Version < 2.0.0 | Version 2.0.0 |
| :--- | :--- | :--- |
| **Initialization** | `ae = AutoEmulate()` followed by `ae.setup(X, y)` | `ae = AutoEmulate(X, y)` (starts comparison immediately) |
| **Model Selection** | `models` argument in `setup()` | `models` argument in `__init__` |
| **Comparison** | Explicit `ae.compare()` call | Implicitly called during initialization |
| **Results Access** | `ae.compare()` returns best model | `ae.best_result().model` or `ae.get_result(id)` |
| **Sensitivity Analysis** | `ae.sensitivity_analysis()` method | Separate `SensitivityAnalysis` class in `autoemulate.core.sensitivity_analysis` |
| **Plotting** | `plot_cv`, `plot_eval` methods | `plot`, `plot_preds`, `plot_surface`, `plot_calibration` methods |
| **Device Support** | CPU primarily | Explicit `device` support (e.g., "cuda", "cpu") |

## Detailed Migration

### 1. Initialization and Setup

**Old Way:**
```python
from autoemulate import AutoEmulate
ae = AutoEmulate()
ae.setup(X, y, param_search=True, test_set_size=0.2)
best_model = ae.compare()
```

**New Way (v2.0.0):**
```python
from autoemulate import AutoEmulate

# Comparison runs immediately upon initialization
ae = AutoEmulate(
    X,
    y,
    n_iter=20,          # Number of parameter search iterations
    n_splits=5,         # Number of CV splits
    test_data=None      # Optional tuple (X_test, y_test)
)
```

### 2. Accessing Results

In v2.0.0, the `AutoEmulate` object holds a collection of results.

```python
# Get the best result (wrapper object)
best_result = ae.best_result()

# Get the best model (trained emulator)
best_model = best_result.model

# Get the name of the best model
model_name = best_result.model_name

# Access metrics (dict of Metric object -> (mean, std))
for metric, score in best_result.test_metrics.items():
    print(f"{metric.name}: {score[0]:.4f} (+/- {score[1]:.4f})")
```

### 3. Sensitivity Analysis

Sensitivity analysis is now decoupled from the main `AutoEmulate` class.

**Old Way:**
```python
ae.sensitivity_analysis()
```

**New Way (v2.0.0):**
```python
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis

# Initialize with the trained model and data/problem definition
sa = SensitivityAnalysis(ae.best_result().model, X)

# Run analysis (Sobol or Morris)
results_df = sa.run(method="sobol", n_samples=1024)

# Plot results
sa.plot_sobol(results_df)
```

### 4. Saving and Loading

**Saving:**
```python
# Save the best model
ae.save(ae.best_result(), path="best_model.joblib")
```

**Loading:**
```python
# Load a model
model = ae.load_model("best_model.joblib")

# Or load a result object
result = ae.load("best_model.joblib")
```

## Complete Example

Here is a complete example demonstrating the v2.0.0 workflow using an epidemic simulation.

```python
import torch
from autoemulate import AutoEmulate
from autoemulate.simulations.epidemic import Epidemic
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis

def main():
    # 1. Generate Data using built-in Simulation
    sim = Epidemic()
    X = sim.sample_inputs(50)  # Generate 50 input samples
    y, X = sim.forward_batch(X) # Run simulation (returns y and valid X)

    # 2. Initialize AutoEmulate and Run Comparison
    print("Running AutoEmulate comparison...")
    ae = AutoEmulate(
        X,
        y,
        n_iter=10,       # Hyperparameter search iterations
        n_splits=3,      # Cross-validation folds
        device="cpu"     # Use "cuda" for GPU if available
    )

    # 3. Inspect Results
    best_result = ae.best_result()
    print(f"\nBest Model: {best_result.model_name}")

    print("Test Metrics:")
    for metric, score in best_result.test_metrics.items():
        print(f"  {metric.name}: {score[0]:.4f} (+/- {score[1]:.4f})")

    # 4. Plotting
    # Plot predicted vs actual
    ae.plot_preds(best_result)

    # 5. Sensitivity Analysis
    print("\nRunning Sensitivity Analysis...")
    sa = SensitivityAnalysis(best_result.model, X)
    sa_results = sa.run(method="sobol", n_samples=512)

    print("\nTop parameters (Sobol Total Order):")
    print(sa_results[sa_results["index"] == "ST"].sort_values("value", ascending=False).head())

    # Plot sensitivity analysis
    sa.plot_sobol(sa_results)

if __name__ == "__main__":
    main()
```
