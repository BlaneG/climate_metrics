# Climate metrics
`climate_metrics` implements standard greenhouse gas methods including Global Warming Potential (GWP) and Average Global Temperature Change Potential (GTP).  The implementation provides flexibility for computing climate metrics for a time series of GHG emissions (e.g., an emissions scenario). 

You can find the documentation [here](https://blaneg.github.io/climate_metrics/), and a quick demonstration of how to use the package [here](notebooks/A-motivating-example.ipynb).


## Installation
`pip install climate_metrics`

Or directly from source:
`pip install git+https://github.com/BlaneG/climate_metrics`

## Development
Tests can be executed on the local copy of climate_metrics by running `python -m pytest` from the root directry. `-m` adds the current directory to `sys.path`.

Notebooks can be checked using nbval: `pytest --nbval notebooks` will check that the notebook outputs are the same and `pytest --nbval-lax notebooks` which check that the notebooks execute without errors.
