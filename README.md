# CSOP Traditional Learning

Research project investigating the use of machine learning to predict objective-function values for candidate solutions in the **Connected Subgraph Optimization Problem (CSOP)**.

The project focuses on using structural graph features to estimate the objective value of candidate subgraphs, with the long-term goal of supporting graph optimization algorithms by reducing the need for expensive objective-function evaluations.

## About the Project

The problem considered is not simply predicting a property of a complete graph.

Given a complete graph \(G\), the optimization problem considers candidate subgraphs:

```text
G
│
├── S1
├── S2
├── S3
└── ...
```

Each candidate subgraph has an objective value.

The first objective function used in the project is the **Triangle Densest Subgraph (TDS)**:

$$
TDS(S) = \frac{T(S)}{|V(S)|}
$$

where \(T(S)\) is the number of triangles in subgraph \(S\).

The number of triangles is calculated using NetworkX's triangle counting functionality:

```python
num_triangles = sum(nx.triangles(G_sub).values()) // 3
```

The machine learning problem is therefore formulated as:

```text
Complete graph G
       +
Candidate subgraph S
       ↓
Feature representation X(G, S)
       ↓
Regression model
       ↓
Predicted TDS(S)
```

The first implementation is intentionally exploratory. The initial goal is to validate the complete pipeline and architecture rather than immediately obtain highly accurate predictions.

---

## Initial Pipeline

The current dataset-generation pipeline is:

```text
Graph Instance
      │
      ▼
Graph Loader
      │
      ▼
Complete Graph G
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
Global Features       Subgraph Generation
                             │
                     ┌───────┼───────┐
                     ▼       ▼       ▼
                    S1      S2      S3
                     │       │       │
                     └───────┼───────┘
                             │
                             ▼
                      Local Features
                             │
                             ▼
                          Pooling
                             │
                             ▼
                     Feature Vector
                             │
                             ▼
                     Objective (TDS)
                             │
                             ▼
                          Sample
                             │
                             ▼
                          Dataset
```

After the dataset has been constructed:

```text
Dataset
   ↓
Training
   ↓
LightGBM
   ↓
Prediction
   ↓
Evaluation
```

---

## Feature Representation

Features have two independent classifications.

### Scope

A feature can be:

* **Global** — computed from the complete graph.
* **Local** — computed from a candidate subgraph.
* **Global and local** — applicable in both contexts.

Global features can be computed once for a complete graph and reused for all candidate subgraphs when appropriate.

### Structural level

A feature can also operate at:

* **Graph level**
* **Node level**
* **Edge level**

These two classifications are independent.

For example:

```text
Global + Graph-level
Global + Node-level
Global + Edge-level

Local + Graph-level
Local + Node-level
Local + Edge-level
```

Node-level and edge-level features can subsequently be aggregated using pooling strategies such as:

```text
Mean
Max
Min
Standard deviation
```

This produces a fixed-size representation suitable for machine learning.

---

## Dataset Samples

Each candidate subgraph becomes one dataset sample.

Conceptually:

```text
(G, S1) → X1 → TDS(S1)
(G, S2) → X2 → TDS(S2)
(G, S3) → X3 → TDS(S3)
```

A sample contains:

```text
graph_id
subgraph_id
features
target
```

For example:

```python
Sample(
    graph_id="graph_001",
    subgraph_id=3,
    features={
        "num_vertices": 100,
        "num_edges": 250,
        "degree_mean": 5.4
    },
    target=0.37
)
```

The `target` is the real objective-function value of the candidate subgraph.

---

## Architecture

The project is organized into independent modules:

```text
src/
│
├── graph/
│   ├── graph.py
│   └── loader.py
│
├── features/
│   ├── base.py
│   ├── engine.py
│   ├── global_features.py
│   ├── local_features.py
│   └── pooling.py
│
├── subgraphs/
│   ├── base.py
│   └── networkx_generator.py
│
├── objectives/
│   ├── base.py
│   └── tds.py
│
├── dataset/
│   ├── sample.py
│   └── builder.py
│
├── model/
│   └── ...
│
├── evaluation/
│   └── ...
│
└── pipeline/
    └── ...
```

The main responsibilities are:

| Module       | Responsibility                             |
| ------------ | ------------------------------------------ |
| `graph`      | Graph representation and loading           |
| `features`   | Feature definitions and feature extraction |
| `subgraphs`  | Candidate subgraph generation              |
| `objectives` | Objective-function implementations         |
| `dataset`    | Dataset sample and dataset construction    |
| `model`      | Machine learning models                    |
| `evaluation` | Prediction evaluation                      |
| `pipeline`   | High-level orchestration                   |

Detailed architectural decisions are documented in [`docs/architecture.md`](docs/architecture.md).

---

## Design Principles

The architecture emphasizes:

* **Modularity** — each component has a well-defined responsibility.
* **Low coupling** — components communicate through clear interfaces.
* **Experimentability** — features, pooling strategies, subgraph generators, objectives, and models can be replaced independently.
* **Reusability** — global features can be computed once and reused across candidate subgraphs.
* **Extensibility** — new feature levels, feature scopes, objectives, and models can be added without redesigning the entire system.
* **Separation of concerns** — dataset construction, training, prediction, and evaluation are separate stages.

---

## Design Patterns

The architecture primarily uses the **Strategy** pattern.

Examples include:

```text
Feature
   ├── NumVerticesFeature
   ├── NumEdgesFeature
   ├── DegreeFeature
   └── ...

Pooling
   ├── MeanPooling
   ├── MaxPooling
   ├── MinPooling
   └── StdPooling

SubgraphGenerator
   └── NetworkXSubgraphGenerator

Predictor
   └── LightGBMPredictor
```

The **Decorator** pattern may be introduced when additional behavior such as caching, timing, normalization, or logging needs to be added without modifying existing components.

A Factory may also be introduced later if the number of experimental configurations makes component creation more complex.

Patterns are used only when they provide a concrete architectural benefit.

---

## Input Data

Each graph instance is stored in its own directory:

```text
instance/
├── metadata
└── adjlist
```

The `metadata` file contains:

```text
number_of_vertices number_of_edges
```

The `adjlist` file contains one line per vertex.

Each adjacency entry has the format:

```text
neighbor_id,edge_id
```

Multiple entries are separated by spaces.

For example:

```text
1,0 3,1 5,2
```

The same `edge_id` appears in the adjacency lists of both endpoints of an undirected edge.

---

## Initial Experiment

The first experiment intentionally uses a small set of components.

### Global features

```text
- number of vertices
- number of edges
```

### Local features

```text
- node degree
```

### Pooling

```text
- mean
- maximum
- minimum
- standard deviation
```

### Subgraph generation

The first experimental implementation uses NetworkX-based subgraph generation, including random path generation.

### Objective

```text
Triangle Densest Subgraph (TDS)
```

### Machine learning model

```text
LightGBM
```

The first objective is to verify that the entire process works:

```text
Complete Graph
      ↓
Candidate Subgraphs
      ↓
Feature Extraction
      ↓
Pooling
      ↓
Feature Vectors
      ↓
TDS Evaluation
      ↓
Dataset
      ↓
LightGBM
      ↓
Prediction
      ↓
Error
```

The initial experiment is not intended to establish the final feature representation or achieve high predictive accuracy.

---

## Testing

The project uses automated tests for the main components.

Current tests include:

```text
tests/
├── test_graph.py
├── test_loader.py
├── test_features.py
├── test_pooling.py
├── test_engine.py
├── test_networkx_generator.py
├── test_tds.py
├── test_sample.py
└── test_builder.py
```

The tests are designed to verify individual components as well as the orchestration performed by the dataset builder.

Run the complete test suite with:

```bash
pytest -v
```

---

## Future Extensions

The architecture is designed to support future experiments involving:

* additional graph-level features;
* additional node-level features;
* edge-level features;
* features computed globally and locally;
* more sophisticated pooling strategies;
* alternative subgraph-generation strategies;
* alternative objective functions;
* alternative regression models;
* feature caching;
* more detailed evaluation;
* integration with graph optimization algorithms;
* integration with a Java-based optimization component.

A possible future division is:

```text
                         JAVA
              ┌─────────────────────┐
              │ Graph Optimization   │
              │                     │
              │ Solution Generation  │
              │ Exploration          │
              └──────────┬──────────┘
                         │
                         │ Interface
                         ▼
                       PYTHON
              ┌─────────────────────┐
              │ Feature Extraction  │
              │ Pooling             │
              │ ML Model            │
              │ Prediction          │
              └──────────┬──────────┘
                         │
                         │ Predicted value
                         ▼
                         JAVA
```

The Java integration is not part of the current implementation. The current priority is to validate the architecture and experimental pipeline in Python.

---

## Project Status

**Status:** In development.

The project is currently focused on implementing and validating the first experimental dataset-generation pipeline, including graph loading, candidate subgraph generation, feature extraction, pooling, TDS evaluation, and dataset construction.