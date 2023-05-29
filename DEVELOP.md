### Run Tests with Python Unittest Module

Run all tests in directory
```bash
python -m unittest discover tests
```

Run all tests from specific file.
```bash
python -m unittest tests.test_gmf_c
```

Run all tests in specific TestCase.
```bash
python -m unittest tests.test_gmf_c.Test_Gmf_C
```

Run specific tests from TestCase.
```bash
python -m unittest tests.test_gmf_c.Test_Gmf_C.test_gmf
```

### Run testfile as Python script

Run all tests in directory
```bash
python tests
```

Run all tests from specific file.
```bash
python tests/test_gmf_c.py
```

### linting with flake8

TODO

### formatting with black

TODO

### pytest

To run entire suite
```bash
pytest
```

To run specific sub-tests
```bash
pytest tests/test_gmf_c.py::Test_Gmf_C::test_gmf
```