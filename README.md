# Quoi

Toolkit for data analysis. Includes methods for visualizing and cleaning data, as well as a custom chart theme.

## Requirements

- Polars
- Plotly

## Features

Features of this toolkit include:
 - Fill missing data (add missing breakdown combinations and fill based on other column);
 - Anomaly detection based on Z-score and IQR;
 - Common chart adjustments:
    - Replace legend entries;
    - Trim axis labels;
    - Summary for each legend entry.

## Running the test suite

Tests can either be run locally with:

```
make test
```

or in a Docker environment with:

```
make docker
```
