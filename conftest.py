
def pytest_addoption(parser):
    """Add a custom command-line option to enable slow tests."""
    parser.addoption(
        "--run-slow", 
        action="store_true", 
        default=False, 
        help="Run tests marked as slow"
    )

def pytest_configure(config):
    """Register the custom marker."""
    config.addinivalue_line("markers", "slow: mark test as slow")