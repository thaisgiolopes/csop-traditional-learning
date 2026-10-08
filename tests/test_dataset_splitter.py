import pytest

from src.dataset.dataset import Dataset
from src.dataset.sample import Sample
from src.dataset.splitter import split_dataset_three_way


def make_dataset(size: int = 50) -> Dataset:
    return Dataset(
        [
            Sample(
                graph_id="graph",
                subgraph_id=index,
                features={"value": float(index)},
                target=float(index),
            )
            for index in range(size)
        ]
    )


def test_three_way_split_sizes_and_disjoint_samples():
    dataset = make_dataset()

    train, validation, test = split_dataset_three_way(
        dataset,
        test_size=0.2,
        validation_size=0.25,
        random_state=17,
    )

    assert train.num_samples == 30
    assert validation.num_samples == 10
    assert test.num_samples == 10

    train_ids = set(train.subgraph_ids)
    validation_ids = set(validation.subgraph_ids)
    test_ids = set(test.subgraph_ids)

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(dataset.subgraph_ids)


def test_three_way_split_is_reproducible():
    dataset = make_dataset()

    first = split_dataset_three_way(
        dataset,
        test_size=0.2,
        validation_size=0.25,
        random_state=17,
    )
    second = split_dataset_three_way(
        dataset,
        test_size=0.2,
        validation_size=0.25,
        random_state=17,
    )

    assert [part.subgraph_ids for part in first] == [
        part.subgraph_ids for part in second
    ]


def test_invalid_split_sizes_are_rejected():
    dataset = make_dataset(size=2)

    with pytest.raises(ValueError):
        split_dataset_three_way(
            dataset,
            test_size=0.5,
            validation_size=0.5,
            random_state=1,
        )