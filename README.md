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
model = sersic_model()
model.fit(mag_range=[22., 23.5])
I, R, n, q = model.sample(mag)
```

See [demo](./demo.ipynb) for more details
