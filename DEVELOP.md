
### linting with flake8

TODO

### formatting with black

TODO

### pytest

To run entire suite
```bash
pytest
```
To run specific file
```bash
pytest tests/test_gmf.py
```

To run specific TestCase, within file
```bash
pytest tests/test_gmf.py::TestGMF
```

To run specific test, within TestCase, within file.
```bash
pytest tests/test_gmf.py::TestGMF::test_gmf
```