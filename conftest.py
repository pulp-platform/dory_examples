import pytest
from local_env import local_env, targets


def pytest_addoption(parser):
    parser.addoption("--target", default=targets[0], choices=targets)
    parser.addoption("--perf", default=False, action="store_true")


@pytest.fixture
def target(request):
    return request.config.getoption("--target")


@pytest.fixture
def env(request):
    return local_env(request.config.getoption("--target"))


@pytest.fixture
def perf(request):
    return request.config.getoption("--perf")
