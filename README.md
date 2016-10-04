# Extrapolation of COSMOS Sersic fits

This package contains a simple code to fit a parametric model to the Sersic fits
from Lackner & Gunn (2012, MNRAS, 421, 2277). The model can then be used to
extrapolate beyond the available magnitude range.

## Requirements

- GalSim
- Scipy
- Matplotlib

## Usage

```python
from cosmos_extrapolation import sersic_model
model = sersic_model()           # Loads necessary COSMOS data
model.fit(mag_range=[22., 23.5]) # Fits the model in the specified range
mag = model.cat['mag_auto']      # Retrieving magnitudes from the COSMOS catalog
I, R, n, q = model.sample(mag)   # Sample intensity, half-light radius, sersic index,
                                 # axis ratio for input magnitude
```

See [demo](./demo.ipynb) for more details
