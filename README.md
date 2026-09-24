# CSOP Traditional Learning

Research project investigating the use of machine learning to predict objective-function values for  andidate solutions in the **Connected Subgraph Optimization Problem (CSOP)**.

The project uses structural graph features to estimate the objective value of candidate subgraphs. In the long term, these predictions may support graph optimization algorithms by reducing the number of expensive direct objective evaluations.

## Project Overview

The problem is not simply to predict a property of a single graph.

Given an input graph \(G\), the optimization problem considers multiple candidate subgraphs:

```text
Input graph G
│
├── Candidate subgraph S1
├── Candidate subgraph S2
├── Candidate subgraph S3
└── ...
```

Each candidate subgraph has an objective-function value. The machine learning model receives a representation of the input graph and a candidate subgraph, then predicts the objective value of that candidate:

$$
X(G, S) \rightarrow \widehat{f(S)}
$$

The first objective function implemented in the project is the **Triangle Densest Subgraph (TDS)** objective:

$$
TDS(S) = \frac{T(S)}{|V(S)|}
$$

where:

- \(T(S)\) is the number of triangles in candidate subgraph \(S\);
- \(|V(S)|\) is the number of vertices in \(S\).

The number of triangles is calculated using NetworkX:

```python
triangle_counts = nx.triangles(graph)
num_triangles = sum(triangle_counts.values()) // 3
```

The initial implementation is exploratory. Its primary goal is to validate the architecture and the complete data-processing pipeline before optimizing prediction quality.

---

## Current Status

The current implementation supports the following workflow:

```text
Raw graph instance
        ↓
Graph loading
        ↓
Candidate subgraph generation
        ↓
Feature extraction
        ↓
Feature pooling
        ↓
TDS objective calculation
        ↓
Dataset construction
        ↓
CSV persistence
        ↓
Dataset loading
        ↓
LightGBM training
        ↓
Prediction
        ↓
Evaluation
```

The current experiment is an integration experiment rather than a final machine learning benchmark.

At this stage:

- candidate subgraphs are generated from random paths;
- only one graph instance is used by the integration experiment;
- the same dataset is used for training and prediction;
- no train/test split or cross-validation is performed;
- the results should not be interpreted as evidence of model generalization.

---

## Installation

Install the project dependencies from the repository root:

```bash
pip install -r `requirements.txt`
```

The main dependencies include:

- NetworkX for graph representation and subgraph generation;
- pandas and NumPy for tabular data processing;
- scikit-learn for regression metrics;
- LightGBM for model training;
- PyTorch Geometric for future graph-learning experiments;
- pytest for automated testing.

---

## Running the Integration Experiment

The complete exploratory pipeline can be executed with:

```bash
python `first_test.py`
```

The script performs the following steps:

1. loads the graph instance from `data/raw/graph_test`;
2. generates candidate connected subgraphs;
3. extracts global and local features;
4. applies pooling strategies to node-level features;
5. calculates the TDS value for each candidate subgraph;
6. builds and persists a tabular dataset;
7. reloads the persisted dataset;
8. trains a LightGBM regression model;
9. generates predictions;
10. calculates evaluation metrics.

The processed dataset is written to:

```text
`integration_test_graph.csv`
```

The experiment output is written to:

```text
`result_first_test.txt`
```

The integration experiment can also be used as an executable demonstration of the complete architecture.

---

## Input Graph Format

Each graph instance is stored in its own directory:

```text
instance/
├── metadata
└── adjlist
```

### Metadata

The `metadata` file contains two integers:

```text
number_of_vertices number_of_edges
```

For example:

```text
5 6
```

### Adjacency List

The `adjlist` file contains one line per vertex.

Each adjacency entry uses the format:

```text
neighbor_id,edge_id
```

Multiple entries are separated by spaces:

```text
1,0 3,1 5,2
```

The same edge identifier must appear in the adjacency lists of both endpoints of an undirected edge.

The `GraphLoader` validates:

- that the required files exist;
- that the number of adjacency-list lines matches the number of vertices;
- that vertex identifiers are valid;
- that self-loops are not present;
- that edge identifiers are consistent;
- that the number of loaded edges matches the metadata.

---

## Feature Representation

Features have two independent classifications.

### Feature Scope

The scope identifies which graph is used during computation.

#### Global features

Global features are calculated from the input graph and can be reused for all candidate subgraphs derived from that graph.

Examples:

```text
Input graph
│
└── Global feature
```

#### Local features

Local features are calculated independently for each candidate subgraph.

```text
Input graph
│
├── S1 → Local feature
├── S2 → Local feature
└── S3 → Local feature
```

### Feature Level

The structural level identifies the entities associated with the feature values:

- graph-level;
- node-level;
- edge-level.

These dimensions are independent. A feature can therefore be:

```text
Global + Graph-level
Global + Node-level
Global + Edge-level

Local + Graph-level
Local + Node-level
Local + Edge-level
```

Node-level and edge-level features can be converted into fixed-size feature vectors using pooling strategies.

---

## Features Currently Implemented

The current integration experiment uses the following features.

### Global graph-level features

- `num_vertices`: number of vertices in the input graph;
- `num_edges`: number of edges in the input graph.

### Local node-level feature

- `degree`: degree of each vertex in the candidate subgraph.

The local degree feature is transformed into the following pooled features:

```text
degree_mean
degree_max
degree_min
degree_std
```

Therefore, the current dataset contains these feature columns:

```text
num_vertices
num_edges
degree_mean
degree_max
degree_min
degree_std
```

Global feature values are cached by `FeatureEngine` and reused across candidate subgraphs from the same input graph.

---

## Pooling Strategies

The current pooling strategies are:

- mean;
- maximum;
- minimum;
- standard deviation.

For example, a node-level degree representation:

```python
{
    0: 2,
    1: 3,
    2: 1
}
```

can be converted into graph-level values such as:

```text
degree_mean = 2.0
degree_max  = 3.0
degree_min  = 1.0
degree_std  = ...
```

This produces a fixed-size numerical representation suitable for regression models.

---

## Candidate Subgraph Generation

The current implementation uses
`NetworkXSubgraphGenerator`.

Candidate subgraphs are generated from random paths in the input graph:

- `num_subgraphs` controls the requested number of paths;
- `path_length` controls the maximum path length;
- `seed` makes the generation reproducible.

Each generated path is converted into a vertex set, and the induced subgraph is created using NetworkX.

Example configuration:

```python
subgraph_generator = NetworkXSubgraphGenerator(
    num_subgraphs=8,
    path_length=3,
    seed=42,
)
```

The generator returns connected candidate subgraphs whenever valid candidates can be produced.

Alternative subgraph-generation strategies can be added through the `SubgraphGenerator` abstraction without changing the dataset-building logic.

---

## Dataset Representation

Each candidate subgraph becomes one `Sample`.

A sample contains:

```text
graph_id
subgraph_id
features
target
```

Example:

```python
Sample(
    graph_id="graph_001",
    subgraph_id=3,
    features={
        "num_vertices": 100,
        "num_edges": 250,
        "degree_mean": 5.4,
        "degree_max": 9.0,
        "degree_min": 1.0,
        "degree_std": 2.1,
    },
    target=0.37,
)
```

The `target` is the true TDS value of the candidate subgraph.

The `Dataset` class converts ordered samples into:

- a feature matrix `X`;
- a target vector `y`;
- graph identifiers;
- subgraph identifiers;
- the original samples.

The dataset validates that:

- at least one sample exists;
- all samples have the same feature schema;
- all feature values are numerical and finite;
- all target values are numerical and finite.

---

## Processed CSV Format

The persisted dataset contains metadata, feature columns, and the target:

```text
graph_id,subgraph_id,num_vertices,num_edges,degree_mean,degree_max,degree_min,degree_std,target
```

Example:

```text
integration_test_graph,0,5,6,1.0,1.0,1.0,0.0,0.0
integration_test_graph,1,5,6,2.0,2.0,2.0,0.0,0.3333333333333333
```

The `DatasetLoader` reconstructs `Sample` objects from the CSV without loading the original graph or recalculating features.

---

## Machine Learning Model

The current model is `LightGBMModel`, an adapter around LightGBM's scikit-learn-compatible `LGBMRegressor`.

Example configuration:

```python
model = LightGBMModel(
    n_estimators=5,
    learning_rate=0.1,
    num_leaves=5,
    min_child_samples=1,
    verbosity=-1,
    random_state=42,
)
```

The model receives the prepared feature matrix and target vector:

```text
Dataset.X → model.fit(...)
Dataset.X → model.predict(...)
```

Feature extraction, preprocessing, dataset splitting, and evaluation are handled outside the model class.

---

## Evaluation

The evaluation pipeline currently calculates:

- **MAE**: Mean Absolute Error;
- **RMSE**: Root Mean Squared Error;
- **R²**: coefficient of determination.

The result is represented by `EvaluationResult` and includes:

```text
mae
rmse
r2
num_samples
```

Example:

```text
Evaluation metrics:
MAE: ...
RMSE: ...
R2: ...
```

### Evaluation limitation

The current integration experiment trains and evaluates the model on the same dataset. This is intentional for the initial architecture validation, but it does not provide a reliable estimate of performance on unseen data.

Future experiments should include:

- separate training and test datasets;
- validation across different graph instances;
- cross-validation;
- evaluation of ranking quality among candidate subgraphs;
- comparison with direct objective-function evaluations.

---

## Project Structure

```text
.
├── data/
│   ├── processed/
│   │   └── `integration_test_graph.csv`
│   └── raw/
│       └── graph_test/
│           ├── adjlist
│           └── metadata
├── docs/
│   └── `architecture.md`
├── experiments/
│   ├── `first_test.py`
│   └── results/
│       └── `result_first_test.txt`
├── src/
│   ├── dataset/
│   │   ├── `builder.py`
│   │   ├── `dataset.py`
│   │   ├── loader.py
│   │   └── `sample.py`
│   ├── evaluation/
│   │   ├── `evaluator.py`
│   │   ├── `metrics.py`
│   │   └── `result.py`
│   ├── features/
│   │   ├── base.py
│   │   ├── `engine.py`
│   │   ├── `graph_features.py`
│   │   ├── `node_features.py`
│   │   └── `pooling.py`
│   ├── graph/
│   │   ├── graph.py
│   │   └── loader.py
│   ├── model/
│   │   ├── base.py
│   │   ├── `lightgbm_model.py`
│   │   └── predictor.py
│   ├── objectives/
│   │   ├── base.py
│   │   └── `tds.py`
│   ├── pipeline/
│   │   ├── `dataset_pipeline.py`
│   │   ├── `prediction_pipeline.py`
│   │   └── `training_pipeline.py`
│   └── subgraphs/
│       ├── base.py
│       └── `networkx_generator.py`
├── tests/
├── `README.md`
├── `requirements.txt`
└── ...
```

Detailed architectural decisions are documented in [docs/architecture.md](docs/architecture.md).

---

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `graph` | Graph representation and input-file loading |
| `features` | Feature definitions, feature extraction, and pooling |
| `subgraphs` | Candidate subgraph generation |
| `objectives` | Objective-function implementations |
| `dataset` | Samples, dataset construction, and CSV loading |
| `model` | Generic model interfaces, LightGBM, and prediction |
| `evaluation` | Regression metrics and evaluation results |
| `pipeline` | High-level orchestration of the complete workflow |

The main components are intentionally independent:

```text
GraphLoader
    ↓
Graph
    ├── FeatureEngine
    ├── SubgraphGenerator
    └── ObjectiveFunction
             ↓
       DatasetBuilder
             ↓
          Dataset
             ↓
       TrainingPipeline
             ↓
          Model
             ↓
     PredictionPipeline
             ↓
        Evaluation
```

---

## Design Principles

The architecture emphasizes:

- **Modularity**: each component has a focused responsibility;
- **Low coupling**: components communicate through explicit interfaces;
- **Experimentability**: features, pooling, generators, objectives, and models
  can be replaced independently;
- **Reusability**: global features can be cached and reused;
- **Extensibility**: new feature levels, objectives, models, and generation
  strategies can be added;
- **Separation of concerns**: dataset construction, training, prediction, and
  evaluation are separate stages.

The project primarily uses the Strategy pattern through abstractions such as:

- `Feature`;
- `Pooling`;
- `SubgraphGenerator`;
- `ObjectiveFunction`;
- `Model`.

---

## Testing

The project includes automated tests for:

- graph representation and loading;
- dataset construction and loading;
- feature extraction;
- feature-engine caching;
- pooling strategies;
- candidate subgraph generation;
- TDS calculation;
- LightGBM training;
- prediction;
- evaluation;
- pipeline integration.

Run the complete test suite with:

```bash
pytest -v
```

The main test files include:

```text
tests/
├── `test_builder.py`
├── test_dataset.py
├── test_dataset_loader.py
├── `test_engine.py`
├── test_evaluator.py
├── test_features.py
├── test_graph.py
├── test_graph_loader.py
├── test_integration.py
├── test_lightgbm_model.py
├── test_networkx_generator.py
├── test_pooling.py
├── `test_predictor.py`
├── test_sample.py
└── test_tds.py
```

---

## Project Status

**Status:** In development.

The project is currently focused on implementing and validating the first experimental dataset-generation pipeline, including graph loading, candidate subgraph generation, feature extraction, pooling, TDS evaluation, and dataset construction.