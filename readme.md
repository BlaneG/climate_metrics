# Climate metrics
`climate_metrics` implements standard greenhouse gas methods including Global Warming Potential (GWP) and Average Global Temperature Change Potential (GTP).  The implementation provides flexibility for computing climate metrics for a time series of GHG emissions represented as a time series. You can find a motivating example [here](notebooks/A-motivating-example.ipynb).


# Development
Tests can be executed on the local copy of climate_metrics by running `python -m pytest` from the root directry which adds the current directory to `sys.path`.

Notebooks are checked using nbval: `pytest --nbval notebooks` will check that the notebook outputs are the same and `pytest --nbval-lax notebooks` which check that the notebooks execute without errors.