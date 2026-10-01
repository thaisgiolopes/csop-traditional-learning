from sklearn.model_selection import train_test_split

from .dataset import Dataset


def split_dataset(
    dataset: Dataset,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[Dataset, Dataset]:
    train_samples, test_samples = train_test_split(
        dataset.samples,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return Dataset(train_samples), Dataset(test_samples)