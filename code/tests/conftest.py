import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test performs a live network fetch "
                   "(deselect with '-m \"not network\"')")
    config.addinivalue_line(
        "markers", "slow: test runs a full walk-forward "
                   "(deselect with '-m \"not slow\"')")
