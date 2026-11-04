import pytest

from src.dataset.dataset import Dataset
from src.dataset.sample import Sample
from src.dataset.splitter import (
    split_dataset_by_ordered_groups,
    split_dataset_stratified_kfold,
    split_dataset_three_way,
)


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


def test_three_way_split_preserves_dataset_configuration_metadata():
    metadata = {"feature_set_id": "feature-set-v2"}
    dataset = Dataset(
        [
            Sample(
                graph_id="graph",
                subgraph_id=index,
                features={"value": float(index)},
                target=float(index),
            )
            for index in range(50)
        ],
        metadata=metadata,
    )

    subsets = split_dataset_three_way(
        dataset,
        test_size=0.2,
        validation_size=0.25,
        random_state=17,
    )

    assert [subset.metadata for subset in subsets] == [metadata] * 3


def test_invalid_split_sizes_are_rejected():
    dataset = make_dataset(size=2)

    with pytest.raises(ValueError):
        split_dataset_three_way(
            dataset,
            test_size=0.5,
            validation_size=0.5,
            random_state=1,
        )


def test_stratified_kfold_is_reproducible_disjoint_and_balances_strata():
    dataset = Dataset(
        [
            Sample(
                graph_id="graph",
                subgraph_id=index,
                features={"value": float(index)},
                target=float(index),
                sample_id=f"sample-{index}",
            )
            for index in range(30)
        ]
    )
    strata = [10] * 15 + [20] * 15

    first = split_dataset_stratified_kfold(
        dataset, strata, n_splits=5, random_state=42
    )
    second = split_dataset_stratified_kfold(
        dataset, strata, n_splits=5, random_state=42
    )

    assert len(first) == 5
    test_partitions = []
    for (train, test), (repeat_train, repeat_test) in zip(
        first, second, strict=True
    ):
        assert set(train.subgraph_ids).isdisjoint(test.subgraph_ids)
        assert train.subgraph_ids == repeat_train.subgraph_ids
        assert test.subgraph_ids == repeat_test.subgraph_ids
        test_strata = [strata[index] for index in test.subgraph_ids]
        assert test_strata.count(10) == 3
        assert test_strata.count(20) == 3
        test_partitions.extend(test.subgraph_ids)
    assert sorted(test_partitions) == list(range(30))


def test_stratified_kfold_rejects_stratum_smaller_than_fold_count():
    dataset = make_dataset(size=10)
    with pytest.raises(ValueError, match="Each stratum"):
        split_dataset_stratified_kfold(
            dataset,
            [10] * 4 + [20] * 6,
            n_splits=5,
            random_state=42,
        )


def test_ordered_group_split_trains_only_on_smaller_groups():
    groups = [size for size in (10, 20, 30, 40, 50, 60) for _ in range(4)]
    dataset = Dataset(
        [
            Sample(
                graph_id="graph",
                subgraph_id=index,
                features={"value": float(index)},
                target=float(index),
            )
            for index in range(len(groups))
        ]
    )
    folds = split_dataset_by_ordered_groups(dataset, groups, n_splits=2)

    assert len(folds) == 2
    tested = []
    for train, test in folds:
        train_groups = {groups[index] for index in train.subgraph_ids}
        test_groups = {groups[index] for index in test.subgraph_ids}
        assert max(train_groups) < min(test_groups)
        assert set(train.subgraph_ids).isdisjoint(test.subgraph_ids)
        tested.extend(test.subgraph_ids)
    assert len(tested) == len(set(tested))
    assert {groups[index] for index in tested} == {30, 40, 50, 60}


def test_ordered_group_split_rejects_uneven_spacing():
    groups = [size for size in (10, 20, 40, 50) for _ in range(2)]
    dataset = make_dataset(size=len(groups))
    with pytest.raises(ValueError, match="equally spaced"):
        split_dataset_by_ordered_groups(dataset, groups, n_splits=2)
