# Clock Implementations

This directory contains the implementation of various aging clocks:

## Directory Structure

- `simple_clocks/`: Contains traditional statistical models for aging predictions
  - `linear_clock.py`: Simple linear regression-based clocks
  - `gompertz_clock.py`: Gompertz survival model clocks (e.g., Kuo 2024)
  - `cph_clock.py`: Cox Proportional Hazards model clocks
- `deep_clocks/`: Contains deep learning-based aging clocks
  - `transformer_clock.py`: Transformer-based models (e.g., Galkin 2025)

## Base Classes

All clocks inherit from the `BaseAgingClock` abstract base class, which provides common functionality:
- Loading model coefficients
- Handling data preprocessing with scalers
- Interface for age prediction

## Usage Example

```python
# Using through the factory (recommended)
from proteoclock import ClockFactory

factory = ClockFactory()
kuo_clock = factory.get_clock("kuo_2024", scaler="ukb_scaler")
predictions = kuo_clock.predict_age(protein_data, scaling='standard')

# Direct usage (advanced)
from proteoclock.clocks.simple_clocks import GompertzClock

clock = GompertzClock(
    coef_file="path/to/weights.txt",
    scaler_file="path/to/scaler.pckl"
)
predictions = clock.predict_age(protein_data, age_data, scaling='standard')
```

## Adding New Clocks

To add a new clock implementation:
1. Create a new class inheriting from `BaseAgingClock`
2. Implement the required methods (`load_coefficients` and `predict_age`)
3. Register the clock in the `ClockFactory` for easy access
