pybuild:
    touch pyproject.toml
    uv --verbose sync --all-extras
    cp -r .venv/lib/python3.11/site-packages/form/_core.pyi python/form