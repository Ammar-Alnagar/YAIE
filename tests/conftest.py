"""
Pytest configuration for YAIE tests
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model-name",
        action="store",
        help="Model name to use for testing (default: microsoft/DialoGPT-small)",
    )
    parser.addoption(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate during tests (default: 100)",
        help="Maximum tokens to generate during tests (default: 100)"
    )


@pytest.fixture
def model_name(request):
    return request.config.getoption("--model-name")


@pytest.fixture
def max_tokens(request):
    return request.config.getoption("--max-tokens")
