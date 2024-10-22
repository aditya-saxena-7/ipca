# Instrumented Principal Components Analysis

## Usage

An exemplary use of the `ipca` package is shown below. The data used is the seminal Grunfeld dataset provided by `statsmodels`. Note, the `fit` method takes a panel of data, `X`, with the following columns:

- `entity id` (numeric)
- `time` (numeric)
- the following columns contain characteristics.

Additionally, it takes a series of dependent variables, `y`, of the same length as `X`.

```python
import numpy as np
from statsmodels.datasets import grunfeld

# Load the Grunfeld dataset
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)

# Establish unique IDs to conform with package requirements
N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(), np.arange(1, N + 1)))
data.firm = data.firm.apply(lambda x: ID[x])

# Use multi-index for panel groups
data = data.set_index(['firm', 'year'])
y = data['invest']
X = data.drop('invest', axis=1)

# Call IPCA
from ipca import InstrumentedPCA

regr = InstrumentedPCA(n_factors=1, intercept=False)
regr = regr.fit(X=X, y=y)
Gamma, Factors = regr.get_factors(label_ind=True)
```

## Installation

The latest release can be installed using `pip`:

```bash
pip install ipca
```

The master branch can be installed by cloning the repository and running the setup:

```bash
git clone https://github.com/bkelly-lab/ipca.git
cd ipca
python setup.py install
```

## Documentation

The latest documentation is published [HERE](#).

## Requirements

Running the package requires:

- **Python 3.7+**
- **NumPy (1.19+)**
- **SciPy (1.6+)**
- **Numba (0.53+)**
- **progressbar (2.5+)**
- **joblib (1.0.1+)**

For testing:

- **pandas (1.2.3+)**
- **scikit-learn (0.24+)**
- **pytest (4.3+)**
- **statsmodels (0.11+)**

Note: Python 3.6+ is a hard requirement. Other versions listed are the ones used in the test environment; older versions may work.

## Acknowledgements

The implementation is inspired by the MATLAB code for IPCA made available on Seth Pruitt's website.

## References

Kelly, Pruitt, Su (2017). "[Instrumented Principal Components Analysis](https://ssrn.com/abstract=2941965)" SSRN.

## Contributions

This information is taken from https://github.com/bkelly-lab/ipca
