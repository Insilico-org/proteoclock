# Data Preprocessing Module

This directory contains utilities for preprocessing protein expression data for use with the aging clocks.

## Key Components

### Scalers (`scalers.py`)

The primary component is the `OLINKScaler` class, which handles:

- Conversion between long and wide formats of protein data
- Imputation of missing values
- Data scaling (standard, min-max)
- Saving and loading scaler objects

## Data Formats

### Long Format (Input)

The standard input format is "long format" with the following columns:
- `patient_id`: Identifier for each sample/patient
- `gene_symbol`: Protein/gene name
- `NPX`: Normalized Protein eXpression value

Example:
```
| patient_id | gene_symbol | NPX   |
|------------|-------------|-------|
| SAMPLE_001 | IL6         | 5.23  |
| SAMPLE_001 | TNF         | 4.87  |
| SAMPLE_002 | IL6         | 6.12  |
| SAMPLE_002 | TNF         | 5.34  |
```

### Wide Format (Internal)

For modeling, data is converted to "wide format" where:
- Each row represents a single patient/sample
- Each column represents a protein/gene
- Cell values contain the NPX values

Example:
```
| patient_id | IL6  | TNF  | ... |
|------------|------|------|-----|
| SAMPLE_001 | 5.23 | 4.87 | ... |
| SAMPLE_002 | 6.12 | 5.34 | ... |
```