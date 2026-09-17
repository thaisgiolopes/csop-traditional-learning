import pytest

from src.features.pooling import (
    Pooling,
    MeanPooling,
    MaxPooling,
    MinPooling,
    StdPooling,
)


def create_test_values():
    """
    Create a set of vertex-level feature values used by the tests.

    Returns:
        dict: A mapping from vertex identifiers to feature values.
    """
    return {
        0: 2,
        1: 4,
        2: 6,
        3: 8,
    }


def test_pooling_is_abstract():
    """Test that the Pooling class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Pooling()


def test_mean_pooling_name():
    """Test the name of the mean pooling strategy."""
    pooling = MeanPooling()

    assert pooling.name == "mean"


def test_max_pooling_name():
    """Test the name of the maximum pooling strategy."""
    pooling = MaxPooling()

    assert pooling.name == "max"


def test_min_pooling_name():
    """Test the name of the minimum pooling strategy."""
    pooling = MinPooling()

    assert pooling.name == "min"


def test_std_pooling_name():
    """Test the name of the standard deviation pooling strategy."""
    pooling = StdPooling()

    assert pooling.name == "std"


def test_mean_pooling():
    """Test that mean pooling computes the arithmetic mean correctly."""
    values = create_test_values()
    pooling = MeanPooling()

    result = pooling.compute(values)

    assert result == 5.0


def test_max_pooling():
    """Test that max pooling returns the largest feature value."""
    values = create_test_values()
    pooling = MaxPooling()

    result = pooling.compute(values)

    assert result == 8


def test_min_pooling():
    """Test that min pooling returns the smallest feature value."""
    values = create_test_values()
    pooling = MinPooling()

    result = pooling.compute(values)

    assert result == 2


def test_std_pooling():
    """Test that standard deviation pooling computes the population
    standard deviation correctly.
    """
    values = create_test_values()
    pooling = StdPooling()

    result = pooling.compute(values)

    expected = (5.0 ** 0.5)

    assert result == pytest.approx(expected)


def test_mean_pooling_with_decimal_values():
    """Test mean pooling with decimal feature values."""
    values = {
        0: 1.5,
        1: 2.5,
        2: 4.0,
    }

    pooling = MeanPooling()

    result = pooling.compute(values)

    assert result == pytest.approx(8.0 / 3.0)


def test_max_pooling_with_negative_values():
    """Test max pooling with negative feature values."""
    values = {
        0: -10,
        1: -3,
        2: -7,
    }

    pooling = MaxPooling()

    result = pooling.compute(values)

    assert result == -3


def test_min_pooling_with_negative_values():
    """Test min pooling with negative feature values."""
    values = {
        0: -10,
        1: -3,
        2: -7,
    }

    pooling = MinPooling()

    result = pooling.compute(values)

    assert result == -10


def test_std_pooling_with_identical_values():
    """Test that standard deviation is zero when all values are identical."""
    values = {
        0: 5,
        1: 5,
        2: 5,
        3: 5,
    }

    pooling = StdPooling()

    result = pooling.compute(values)

    assert result == 0.0


def test_pooling_rejects_empty_values():
    """Test that all pooling strategies reject an empty mapping."""
    pooling_strategies = [
        MeanPooling(),
        MaxPooling(),
        MinPooling(),
        StdPooling(),
    ]

    for pooling in pooling_strategies:
        with pytest.raises(ValueError):
            pooling.compute({})


def test_pooling_rejects_non_dictionary_values():
    """Test that all pooling strategies reject non-dictionary inputs."""
    pooling_strategies = [
        MeanPooling(),
        MaxPooling(),
        MinPooling(),
        StdPooling(),
    ]

    invalid_values = [
        [1, 2, 3],
        (1, 2, 3),
        {1, 2, 3},
        "invalid",
        42,
        None,
    ]

    for pooling in pooling_strategies:
        for values in invalid_values:
            with pytest.raises(TypeError):
                pooling.compute(values)


def test_mean_pooling_with_single_value():
    """Test mean pooling with a single vertex-level value."""
    values = {0: 7}

    pooling = MeanPooling()

    assert pooling.compute(values) == 7.0


def test_max_pooling_with_single_value():
    """Test max pooling with a single vertex-level value."""
    values = {0: 7}

    pooling = MaxPooling()

    assert pooling.compute(values) == 7


def test_min_pooling_with_single_value():
    """Test min pooling with a single vertex-level value."""
    values = {0: 7}

    pooling = MinPooling()

    assert pooling.compute(values) == 7


def test_std_pooling_with_single_value():
    """Test that standard deviation is zero for a single value."""
    values = {0: 7}

    pooling = StdPooling()

    assert pooling.compute(values) == 0.0