# Materials Directory

This directory contains essential materials required for the ProteoClock package:

## Directory Structure

- `deep_clocks/`: Weights and configurations for deep learning-based clocks
  - `galkin_2025/`: Transformer-based models for aging prediction
- `scalers/`: Pre-trained scalers for data normalization
  - Contains scalers tailored for specific clocks and datasets
- `simple_clocks/`: Coefficients and parameters for statistical clocks
  - `goeminne_2025/`: OrganAge clocks (chronological and mortality)
  - `kuo_2024/`: Proteomic Aging Clock (PAC) parameters
- `test_data/`: Sample datasets for testing and demonstration

## Usage

These materials are automatically loaded by the appropriate classes and shouldn't need to be accessed directly. The `ClockFactory` handles loading the correct materials for each clock type.

```python
from proteoclock import ClockFactory

# Factory will automatically load the appropriate coefficients and scalers
factory = ClockFactory()
clock = factory.get_clock("kuo_2024", scaler="ukb_scaler")
```

## File Formats

### Coefficient Files (for custom simple clocks)

Simple text files with the following format:
- One protein/gene per line
- Each line contains: `[gene_symbol] [coefficient]`
- Optional header line (will be skipped if present)

You may add custom coefficients to the `simple_clocks/` directory to create a new clock. Check if the module sees them with factory.view_clocks()

Example:
```
gene_symbol coefficient
IL6 0.0234
TNF 0.0187
CRP 0.0456
```

### Scaler Files

Binary pickle files containing custom scaler dictionaries to be loaded by `OLINKScaler` class method.
