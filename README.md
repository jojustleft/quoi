# Quoi

Data analysis toolkit. Includes methods for visualizing and cleaning data, as well as a custom chart theme.

## Requirements

- Polars
- Plotly
- Scikit-learn

## Features

Features of this toolkit include:
 - Fill missing data (add missing breakdown combinations and fill based on other column);
 - Anomaly detection based on Z-score and IQR;
 - Hopkins statistic for measuring data cluster tendency;
 - Common chart adjustments:
    - Replace legend entries;
    - Trim axis labels;
    - Summary for each legend entry;
    - Merge charts into a single figure with dropdown selector.

## Running the test suite

Tests can either be run locally with:

```
make test
```

or in a Docker environment with:

```
make docker
```
