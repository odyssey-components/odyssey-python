# Odyssey Python

- `odyssey-config` in `python/config` provides typed configuration loading and validation.
- `odylab-training` in `python/training` provides shared worker, monitoring and training utilities.

Release `v0.1.0` contains both packages. Install them together with Python 3.14:

```sh
uv pip install \
  'odyssey-config @ git+https://github.com/odyssey-components/odyssey-python.git@v0.1.0#subdirectory=python/config' \
  'odylab-training @ git+https://github.com/odyssey-components/odyssey-python.git@v0.1.0#subdirectory=python/training'
```

Run `uv sync --locked --all-packages` and `uv run --locked pytest -q python/config/tests python/training/tests` for tests.
Build each wheel with `uv build --no-sources --wheel --package <package>`.

Configuration test fixtures were taken from Odyssey at `6d92d8a695a0d5f864f59a832e04077c389ab607`.
