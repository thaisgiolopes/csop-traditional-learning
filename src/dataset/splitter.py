"""Dataset splitting helpers."""

from collections import Counter

from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

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
    return (
        Dataset(train_samples, metadata=dataset.metadata),
        Dataset(test_samples, metadata=dataset.metadata),
    )


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
        Dataset(train_samples, metadata=dataset.metadata),
        Dataset(validation_samples, metadata=dataset.metadata),
        Dataset(test_samples, metadata=dataset.metadata),
    )


def split_dataset_stratified_holdout(
    dataset: Dataset,
    strata: list[int],
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[Dataset, Dataset]:
    """Reserve a reproducible, stratified final test set from a dataset."""
    if len(strata) != dataset.num_samples:
        raise ValueError("strata must contain one value per dataset sample.")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TypeError("test_size must be numeric.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    development_samples, test_samples = train_test_split(
        dataset.samples,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strata,
    )
    return (
        Dataset(development_samples, metadata=dataset.metadata),
        Dataset(test_samples, metadata=dataset.metadata),
    )


def split_dataset_stratified_kfold(
    dataset: Dataset,
    strata: list[int],
    *,
    n_splits: int = 5,
    random_state: int | None = 42,
) -> list[tuple[Dataset, Dataset]]:
    """Create reproducible folds preserving each configured size stratum."""
    if len(strata) != dataset.num_samples:
        raise ValueError("strata must contain one value per dataset sample.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    counts = Counter(strata)
    if not counts or min(counts.values()) < n_splits:
        raise ValueError(
            "Each stratum must contain at least n_splits samples for "
            "stratified K-fold."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    samples = dataset.samples
    folds = []
    for train_indices, test_indices in splitter.split(dataset.X, strata):
        train = Dataset(
            [samples[index] for index in train_indices],
            metadata=dataset.metadata,
        )
        test = Dataset(
            [samples[index] for index in test_indices],
            metadata=dataset.metadata,
        )
        folds.append((train, test))
    return folds


def split_dataset_by_ordered_groups(
    dataset: Dataset,
    groups: list[int],
    *,
    n_splits: int = 5,
) -> list[tuple[Dataset, Dataset]]:
    """Use TimeSeriesSplit over ordered, equally spaced groups such as sizes."""
    if len(groups) != dataset.num_samples:
        raise ValueError("groups must contain one value per dataset sample.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    ordered_groups = sorted(set(groups))
    if len(ordered_groups) <= n_splits:
        raise ValueError("TimeSeriesSplit requires more groups than n_splits.")
    if len(ordered_groups) < 2:
        raise ValueError("At least two distinct groups are required.")
    intervals = {
        current - previous
        for previous, current in zip(ordered_groups, ordered_groups[1:])
    }
    if len(intervals) != 1:
        raise ValueError(
            "Ordered groups must be equally spaced for comparable "
            "TimeSeriesSplit folds."
        )

    group_splitter = TimeSeriesSplit(n_splits=n_splits)
    samples = dataset.samples
    indices_by_group = {
        group: [index for index, value in enumerate(groups) if value == group]
        for group in ordered_groups
    }
    folds = []
    for train_group_indices, test_group_indices in group_splitter.split(
        ordered_groups
    ):
        train_groups = [ordered_groups[index] for index in train_group_indices]
        test_groups = [ordered_groups[index] for index in test_group_indices]
        train_indices = [
            index for group in train_groups for index in indices_by_group[group]
        ]
        test_indices = [
            index for group in test_groups for index in indices_by_group[group]
        ]
        folds.append(
            (
                Dataset([samples[index] for index in train_indices], dataset.metadata),
                Dataset([samples[index] for index in test_indices], dataset.metadata),
            )
        )
    return folds
