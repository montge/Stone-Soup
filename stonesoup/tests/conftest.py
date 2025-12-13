import os

import pytest
from hypothesis import Phase, Verbosity, settings

from ..base import Base, Property

# Hypothesis profile configuration
# CI profile: deterministic, stricter settings for reproducibility
settings.register_profile(
    "ci",
    max_examples=100,
    verbosity=Verbosity.verbose,
    deadline=None,  # Disable deadline in CI for stability
    derandomize=True,  # Make tests deterministic
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    print_blob=True,  # Print reproducer blob on failure
)

# Development profile: faster iteration, less strict
settings.register_profile(
    "dev",
    max_examples=10,
    verbosity=Verbosity.normal,
    deadline=None,
)

# Debug profile: minimal examples for debugging
settings.register_profile(
    "debug",
    max_examples=5,
    verbosity=Verbosity.verbose,
    deadline=None,
    derandomize=True,
)

# Load profile from environment or default to 'dev'
profile = os.environ.get("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(profile)


class _TestBase(Base):
    property_a: int = Property()
    property_b: str = Property()
    property_c: int = Property(default=123)


@pytest.fixture(scope="session")
def base():
    return _TestBase
