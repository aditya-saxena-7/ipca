import numpy as np
from statsmodels.datasets import grunfeld
from ipca import InstrumentedPCA

# Load the Grunfeld dataset as a pandas DataFrame
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)

N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(), np.arange(1, N+1)))
data.firm = data.firm.apply(lambda x: ID[x])

# use multi-index for panel groups
data = data.set_index(['firm', 'year'])

y = data['invest']
X = data.drop('invest', axis=1)

# Initialize the IPCA model with 1 latent factor, and no intercept in the regression
regr = InstrumentedPCA(n_factors=2, intercept=False)

# Fit the model to the data (X as instruments, y as the dependent variable)
regr = regr.fit(X=X, y=y)

# Retrieve the factor loadings (Gamma) and latent factors
Gamma, Factors = regr.get_factors(label_ind=True)

print('Gamma:', Gamma)
print('Factors', Factors)