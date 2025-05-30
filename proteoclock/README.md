# ProteoClock Package

The core implementation of the ProteoClock package for proteomic aging clock analysis.

## Package Structure

- `clocks/`: Clock implementations (simple and deep learning-based)
- `data_preprocess/`: Data handling and preprocessing utilities
- `materials/`: Model coefficients, trained scalers, and test data
- `factory.py`: Factory class for easily initializing different clocks
- `other.py`: Common utility functions, including test data loading

## Key Components

### ClockFactory

The recommended way to initialize aging clocks:

```python
from proteoclock import ClockFactory

# Initialize factory
factory = ClockFactory()

# View available clocks
factory.view_clocks()

# Get a specific clock
clock = factory.get_clock("kuo_2024", scaler="ukb_scaler")

# Or get an organ-specific clock
liver_clock = factory.get_clock(
    "goeminne_2025_full_chrono", 
    subtype="liver", 
    scaler="goeminne_2025_full"
)
```

### Clock Classes

All aging clock implementations inherit from the `BaseAgingClock` abstract base class, providing a consistent interface:

```python
# All clocks support a similar prediction API
predictions = clock.predict_age(protein_data, scaling='standard')
```

## Sample Data

The package includes sample data for testing and demonstration:

```python
from proteoclock import load_test_age_data, load_test_protein_data

# Load sample data
age_data = load_test_age_data()
protein_data = load_test_protein_data()
```

See individual module READMEs for more detailed information on each component.
