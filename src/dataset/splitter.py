"""Dataset splitting helpers."""

from sklearn.model_selection import train_test_split

from .dataset import Dataset


def split_dataset(
    dataset: Dataset,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into training and test datasets."""
    train_samples, test_samples = train_test_split(
        dataset.samples,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return Dataset(train_samples), Dataset(test_samples)


def split_dataset_three_way(
    dataset: Dataset,
    *,
    test_size: float,
    validation_size: float,
    random_state: int | None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return train, validation, and isolated final test datasets.

    The final test set is selected first. validation_size is applied only
    to the remaining development samples, matching train_test_split
    semantics.
    """
    development_samples, test_samples = train_test_split(
        dataset.samples,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_samples, validation_samples = train_test_split(
        development_samples,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    return (
        Dataset(train_samples),
        Dataset(validation_samples),
        Dataset(test_samples),
    )