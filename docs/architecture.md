# Arquitetura do Projeto

## 1. Visão geral

Este projeto investiga o uso de aprendizado de máquina para auxiliar a exploração de soluções no **Connected Subgraph Optimization Problem (CSOP)**.

A ideia central é utilizar características estruturais de grafos para construir um modelo capaz de **predizer o valor da função objetivo de subgrafos candidatos**, reduzindo potencialmente a necessidade de avaliar diretamente a função objetivo em todas as soluções durante um processo de otimização.

A primeira função objetivo utilizada nos experimentos será o **Triangle Densest Subgraph (TDS)**.

O TDS de um subgrafo \(S\) é definido como:

$$
TDS(S) = \frac{T(S)}{|V(S)|}
$$

onde \(T(S)\) representa o número de triângulos presentes no subgrafo.

A contagem de triângulos pode ser obtida utilizando a função `triangles` do NetworkX. Como essa função retorna a quantidade de triângulos incidentes em cada vértice, o número total de triângulos é calculado por:

```python
num_triangles = sum(nx.triangles(G_sub).values()) // 3
```

O objetivo da primeira implementação não é obter imediatamente boas previsões. O foco inicial é construir e validar uma arquitetura que permita experimentar diferentes representações, características, estratégias de pooling e modelos de aprendizado.

---

# 2. Problema de aprendizado

O problema não consiste simplesmente em receber um grafo e predizer o TDS desse grafo.

O problema considerado é:

```text
Grafo completo G
       │
       ├── Subgrafo S1 → TDS(S1)
       ├── Subgrafo S2 → TDS(S2)
       ├── Subgrafo S3 → TDS(S3)
       ├── ...
       └── Subgrafo Sn → TDS(Sn)
```

O objetivo do processo de otimização é encontrar um subgrafo \(S\) que maximize:

$$
TDS(S)
$$

O aprendizado de máquina será utilizado para estimar esse valor para subgrafos candidatos.

Assim, cada exemplo do dataset representa uma relação entre um **grafo completo**, um **subgrafo candidato** e o valor real da função objetivo:

$$
X(G,S) \rightarrow TDS(S)
$$

Essa formulação é importante para a arquitetura, pois características do grafo completo e características do subgrafo possuem papéis diferentes na representação de cada exemplo.

---

# 3. Princípios arquiteturais

A arquitetura é guiada pelos seguintes princípios.

## 3.1. Alta modularidade

Cada componente deve possuir uma responsabilidade bem definida.

Por exemplo:

* `Graph` representa um grafo;
* `GraphLoader` traduz a representação dos arquivos para um `Graph`;
* geradores de subgrafos produzem candidatos;
* características calculam propriedades estruturais;
* `FeatureEngine` coordena a extração;
* `Pooling` agrega características de nós ou arestas;
* `ObjectiveFunction` calcula o valor real da função objetivo;
* `DatasetBuilder` combina todas essas informações;
* o modelo realiza treinamento e predição;
* o avaliador mede o erro.

Nenhum desses componentes deve concentrar responsabilidades pertencentes aos demais.

---

## 3.2. Baixo acoplamento

Os componentes devem depender de abstrações e interfaces sempre que possível.

Uma característica não deve conhecer o modelo de aprendizado.

O modelo não deve conhecer como uma característica foi calculada.

O gerador de subgrafos não deve conhecer o TDS.

O `DatasetBuilder` deve apenas coordenar essas operações.

A relação desejada é:

```text
Graph
  │
  ├── Features
  │
  ├── Subgraph Generator
  │
  └── Objective Function
          │
          ▼
    Dataset Builder
          │
          ▼
    Feature Vector + Target
          │
          ▼
       ML Model
```

---

## 3.3. Facilidade de experimentação

A pesquisa possui caráter exploratório. Portanto, a arquitetura deve permitir testar diferentes hipóteses sem exigir alterações estruturais no sistema.

Por exemplo:

```text
Experimento A
|V| + |E| + média do grau

Experimento B
|V| + |E| + média + máximo do grau

Experimento C
|V| + |E| + características de nós + características de arestas

Experimento D
características globais + características locais

Experimento E
diferentes estratégias de pooling
```

Da mesma forma, o modelo de aprendizado poderá ser substituído futuramente.

---

## 3.4. Separação entre grafo completo e subgrafo

Essa é uma decisão importante da arquitetura atual.

O sistema sempre parte de um **grafo completo** \(G\).

A partir dele são gerados diversos subgrafos candidatos:

$$
S_1, S_2, ..., S_n
$$

As características podem ser calculadas em diferentes contextos.

Por exemplo:

```text
Grafo completo G
│
├── Características globais
│
└── geração dos subgrafos
        │
        ├── S1
        │    └── Características locais
        │
        ├── S2
        │    └── Características locais
        │
        └── S3
             └── Características locais
```

Essa separação permite que características potencialmente mais custosas sejam calculadas apenas uma vez para o grafo completo quando isso fizer sentido.

---

# 4. Duas dimensões das características

As características possuem duas classificações independentes.

## 4.1. Escopo da característica

O escopo indica **sobre qual grafo a característica deve ser calculada**.

### Global

Uma característica global é calculada sobre o grafo completo.

```text
G
│
└── Global Feature
```

O resultado pode ser reutilizado para todos os subgrafos derivados daquele grafo.

Exemplos:

```text
número de vértices
número de arestas
outras propriedades estruturais globais
```

---

### Local

Uma característica local é calculada sobre cada subgrafo candidato.

```text
G
│
├── S1 → Local Feature
├── S2 → Local Feature
└── S3 → Local Feature
```

Isso permite representar propriedades específicas da solução candidata.

---

### Global e local

Algumas características podem ser interessantes nos dois contextos.

Nesse caso, a mesma característica pode possuir uma implementação ou configuração para:

```text
G → característica global

S → característica local
```

A arquitetura não deve assumir que uma característica pertence necessariamente a apenas um dos escopos.

---

# 5. Nível da característica

Independentemente do escopo, uma característica também possui um **nível estrutural**.

Os níveis considerados são:

```text
Graph-level
Node-level
Edge-level
```

Essas duas classificações não devem ser confundidas.

Por exemplo:

```text
                  Nível
               ┌───────────┐
               │           │
Global ────────┼── Graph   │
               │── Node    │
               │── Edge    │
               └───────────┘

Local ─────────┼── Graph
               │── Node
               │── Edge
```

Assim, é possível ter, por exemplo:

```text
Global + Graph-level
Global + Node-level
Global + Edge-level

Local + Graph-level
Local + Node-level
Local + Edge-level
```

Essa separação torna a arquitetura mais flexível para futuras características.

---

# 6. Representação do grafo

O módulo `graph` é responsável exclusivamente pela representação e carregamento dos grafos.

A classe `Graph` representa um grafo não direcionado.

Ela fornece operações estruturais básicas, como:

```text
num_vertices
num_edges
has_vertex()
has_edge()
neighbors()
```

A classe não deve conhecer:

* características;
* pooling;
* objetivos;
* modelos de aprendizado;
* algoritmos de otimização.

---

# 7. Carregamento das instâncias

Cada instância possui uma pasta própria contendo:

```text
instance/
├── metadata
└── adjlist
```

O arquivo `metadata` contém:

```text
num_vertices num_edges
```

O arquivo `adjlist` possui uma linha para cada vértice.

A linha correspondente ao vértice `v` contém entradas no formato:

```text
neighbor_id,edge_id
```

e várias entradas são separadas por espaços.

Por exemplo:

```text
1,0 3,1 5,2
```

O `edge_id` identifica a mesma aresta nas listas de adjacência de suas duas extremidades.

O `GraphLoader` possui uma única responsabilidade:

```text
arquivos
   ↓
Graph
```

Ele não deve calcular características nem objetivos.

---

# 8. Geração de subgrafos

Depois que o grafo completo é carregado, o sistema gera subgrafos candidatos.

A responsabilidade pertence ao módulo `subgraphs`.

Uma abstração:

```python
class SubgraphGenerator:
    def generate(self, graph):
        ...
```

A primeira implementação utiliza NetworkX para realizar uma geração experimental de candidatos.

Entre as funções consideradas está:

```text
networkx.generate_random_paths
```

e operações de construção de subgrafos.

A implementação inicial não pretende representar o processo definitivo de geração de soluções. Ela serve para criar um conjunto inicial de candidatos para construir e testar o pipeline.

A arquitetura deve permitir posteriormente substituir essa estratégia por:

```text
RandomSubgraphGenerator
MetaheuristicSubgraphGenerator
NeighborhoodSubgraphGenerator
JavaSubgraphGenerator
...
```

sem modificar as etapas de extração, objetivo ou aprendizado.

---

# 9. Extração de características

As características são implementadas como estratégias independentes.

Uma abstração conceitual:

```python
class Feature:
    name
    scope
    level

    def compute(context):
        ...
```

O contexto fornece as informações necessárias para a característica, incluindo o grafo completo e o subgrafo quando aplicável.

A característica deve ser responsável apenas por calcular seu próprio valor.

---

# 10. Feature Context

Para evitar que cada característica precise receber vários parâmetros separadamente, o sistema utiliza um contexto.

Conceitualmente:

```text
FeatureContext
│
├── full_graph
└── subgraph
```

Assim, uma característica pode decidir qual representação utilizar de acordo com seu escopo.

Por exemplo:

```text
Global Feature
    ↓
full_graph

Local Feature
    ↓
subgraph
```

Isso também permite que uma característica global e local utilize a mesma infraestrutura.

---

# 11. Feature Engine

O `FeatureEngine` é responsável por coordenar a execução das características.

Ele não implementa as características.

Sua responsabilidade é:

```text
FeatureEngine
      │
      ├── Feature A
      ├── Feature B
      ├── Feature C
      └── Feature D
```

e executar as características apropriadas para determinado contexto.

Por exemplo:

```python
global_results = engine.compute_global(context)
local_results = engine.compute_local(context)
```

Uma vantagem importante dessa separação é permitir que características globais sejam calculadas apenas uma vez.

---

# 12. Reutilização das características globais

Essa decisão é importante para a eficiência da construção do dataset.

Considere:

```text
G
│
├── S1
├── S2
├── S3
├── ...
└── Sn
```

Se uma característica depende somente de `G`, seu cálculo não precisa ser repetido para cada subgrafo.

O fluxo será:

```text
G
│
└── calcular características globais
            │
            ▼
       Global Features
            │
            ├──────────────┐
            │              │
            ▼              ▼
           S1             S2 ...
```

Enquanto características locais serão calculadas individualmente:

```text
S1 → Local Features
S2 → Local Features
S3 → Local Features
...
```

Essa organização reduz cálculos repetidos e mantém clara a diferença entre informações do problema original e informações da solução candidata.

---

# 13. Pooling

Características de nível de nó ou de aresta podem produzir múltiplos valores.

Por exemplo:

```text
graus dos vértices

v0 → 2
v1 → 5
v2 → 3
v3 → 7
```

Esses valores ainda não formam um vetor de tamanho fixo para representar o grafo ou subgrafo.

O pooling transforma esses valores em estatísticas de tamanho fixo.

Exemplos:

```text
MeanPooling
MaxPooling
MinPooling
StdPooling
```

Assim:

```text
degree
  │
  ├── mean
  ├── max
  ├── min
  └── std
```

resulta em:

```text
degree_mean
degree_max
degree_min
degree_std
```

O pooling não deve conhecer a implementação do `Graph`.

Ele recebe os valores produzidos pela característica e os agrega.

---

# 14. Vetor de características

Depois da extração e do pooling, cada par `(G, S)` deve produzir uma representação de tamanho fixo.

Por exemplo:

```python
{
    "num_vertices": 100,
    "num_edges": 250,
    "degree_mean": 5.0,
    "degree_max": 21,
    "degree_min": 1,
    "degree_std": 3.2
}
```

No problema atual, esses valores podem combinar:

```text
Características do grafo completo
+
Características do subgrafo
```

Essa distinção é importante para o aprendizado, pois o modelo precisa receber informações tanto sobre o contexto do problema quanto sobre a solução candidata.

Os nomes das características devem ser preservados até a etapa de construção do dataset, em vez de trabalhar apenas com posições numéricas.

---

# 15. Função objetivo

A função objetivo deve ser independente das características e do modelo de aprendizado.

Uma abstração:

```python
class ObjectiveFunction:
    def compute(self, graph):
        ...
```

A primeira implementação será:

```text
TDSObjective
```

Para um subgrafo \(S\):

$$
TDS(S) = \frac{T(S)}{|V(S)|}
$$

onde:

$$
T(S) =
\frac{\sum_{v \in V(S)} triangles(v)}{3}
$$

O objetivo será calculado sobre cada subgrafo candidato.

```text
S1 → TDS(S1)
S2 → TDS(S2)
S3 → TDS(S3)
...
```

Esses valores constituem o `target` do dataset.

---

# 16. Sample

Cada candidato do dataset é representado por um `Sample`.

Conceitualmente:

```text
Sample
├── graph_id
├── subgraph_id
├── features
└── target
```

Exemplo:

```python
Sample(
    graph_id="graph_001",
    subgraph_id=4,
    features={
        "num_vertices": 100,
        "num_edges": 250,
        "degree_mean": 5.4
    },
    target=0.37
)
```

`Sample` é apenas um objeto de dados.

Ele não calcula características, pooling ou objetivos.

---

# 17. Dataset Builder

O `DatasetBuilder` é a principal camada de orquestração da construção do dataset.

Seu fluxo é:

```text
GraphLoader
     │
     ▼
Grafo completo G
     │
     ├─────────────────────┐
     │                     │
     ▼                     ▼
Global Features     Subgraph Generator
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
            S1            S2            S3
             │             │             │
             ▼             ▼             ▼
        Local Features
             │
             ▼
          Pooling
             │
             ▼
    Feature Vector X(G,S)
             │
             ▼
      Objective Function
             │
             ▼
         TDS(S)
             │
             ▼
          Sample
```

Para cada subgrafo candidato:

$$
(G,S) \rightarrow X(G,S), TDS(S)
$$

O `DatasetBuilder` não deve:

* implementar características;
* implementar pooling;
* implementar geração de subgrafos;
* implementar a função objetivo;
* treinar o modelo.

Sua responsabilidade termina na construção dos `Sample`s.

---

# 18. Dataset

Para um grafo completo \(G\) com candidatos:

$$
S_1, S_2, ..., S_n
$$

o dataset terá:

```text
(G, S1) → X1 → TDS(S1)
(G, S2) → X2 → TDS(S2)
(G, S3) → X3 → TDS(S3)
...
(G, Sn) → Xn → TDS(Sn)
```

Assim, o conjunto de treinamento possui vários exemplos associados ao mesmo grafo completo.

Essa estrutura é diferente de um problema em que cada grafo possui apenas um valor-alvo.

---

# 19. Modelo de aprendizado

O primeiro modelo será o **LightGBM**, utilizado para regressão.

Entretanto, o restante da arquitetura não deve depender diretamente do LightGBM.

Uma abstração futura pode ser:

```python
class Predictor:
    def train(self, X, y):
        ...

    def predict(self, X):
        ...
```

A implementação inicial poderá ser:

```text
Predictor
    │
    └── LightGBMPredictor
```

Posteriormente:

```text
Predictor
    ├── LightGBMPredictor
    ├── RandomForestPredictor
    ├── XGBoostPredictor
    └── ...
```

O objetivo é permitir a substituição do modelo sem modificar a construção das características.

---

# 20. Treinamento

A construção do dataset e o treinamento são etapas diferentes.

Primeiro:

```text
Grafo completo
      ↓
Subgrafos
      ↓
Features
      ↓
Pooling
      ↓
Samples
      ↓
Dataset
```

Depois:

```text
Dataset
   ↓
Train / Validation / Test
   ↓
Modelo
   ↓
Treinamento
```

Essa separação permite gerar o dataset uma vez e experimentar diferentes modelos posteriormente.

---

# 21. Predição

Depois de treinado, o modelo pode receber um novo candidato.

O fluxo conceitual é:

```text
Grafo completo G
       │
       ▼
Subgrafo candidato S
       │
       ▼
Extração de características
       │
       ▼
Pooling
       │
       ▼
Vetor X(G,S)
       │
       ▼
Modelo treinado
       │
       ▼
TDS previsto
```

A predição não substitui a função objetivo real em todas as situações. Nesta etapa da pesquisa, ela será utilizada para investigar se características estruturais podem fornecer uma estimativa útil do valor da função objetivo.

---

# 22. Avaliação

A avaliação compara o valor previsto pelo modelo com o valor real calculado pela função objetivo.

```text
TDS real
   │
   ├──────────► Erro
   │
TDS previsto
```

A primeira avaliação pode utilizar o erro absoluto:

$$
|y-\hat{y}|
$$

Posteriormente poderão ser avaliadas métricas como:

```text
MAE
RMSE
MAPE
R²
```

A escolha definitiva das métricas será feita conforme os experimentos.

---

# 23. Padrões de projeto

A arquitetura utiliza padrões de projeto quando eles resolvem uma necessidade concreta.

## 23.1. Strategy

O **Strategy** é o principal padrão utilizado.

Ele é adequado porque diversas partes do sistema possuem estratégias substituíveis.

Exemplos:

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
   ├── NetworkXSubgraphGenerator
   └── ...

Predictor
   ├── LightGBMPredictor
   └── ...
```

A substituição de uma estratégia não deve exigir alterações nas classes que a utilizam.

---

# 24. Decorator

O **Decorator** é considerado como um padrão complementar.

Ele não deve ser utilizado para representar novas características.

Seu objetivo é permitir adicionar comportamentos a componentes existentes sem modificar suas implementações.

Possíveis aplicações futuras:

```text
Feature
   │
   ├── CacheDecorator
   ├── NormalizationDecorator
   ├── TimingDecorator
   └── LoggingDecorator
```

Por exemplo:

```text
DegreeFeature
      │
      ▼
TimingDecorator
      │
      ▼
FeatureEngine
```

Isso pode ser útil principalmente quando forem investigados custos computacionais das características.

---

# 25. Factory

Uma Factory pode ser introduzida quando o número de configurações experimentais aumentar.

Por exemplo:

```text
Configuration
      │
      ▼
FeatureFactory
      │
      ├── DegreeFeature
      ├── TriangleFeature
      └── ...
```

Neste momento, não é necessário introduzir uma Factory apenas por formalidade.

Ela deve ser adicionada quando a criação dos componentes começar a justificar essa abstração.

---

# 26. Pipeline e Facade

O pipeline funciona como uma camada de orquestração.

Ele não deve implementar os cálculos internos.

Sua responsabilidade é coordenar:

```text
Loader
  ↓
Subgraph Generator
  ↓
Feature Engine
  ↓
Pooling
  ↓
Objective
  ↓
Dataset Builder
  ↓
Model
  ↓
Evaluation
```

Essa separação permite testar cada componente individualmente.

---

# 27. Arquitetura atual

A arquitetura atual pode ser resumida da seguinte forma:

```text
                         Graph Instance
                              │
                              ▼
                        ┌────────────┐
                        │GraphLoader │
                        └─────┬──────┘
                              │
                              ▼
                     Complete Graph G
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       Global Feature Extraction     Subgraph Generator
                │                           │
                │                    ┌──────┼──────┐
                │                    ▼      ▼      ▼
                │                   S1     S2     S3
                │                    │      │      │
                │                    └──────┼──────┘
                │                           │
                │                           ▼
                │                   Local Feature Extraction
                │                           │
                └──────────────┐            │
                               ▼            ▼
                         Feature Results
                               │
                               ▼
                            Pooling
                               │
                               ▼
                      Feature Vector X(G,S)
                               │
                               ▼
                       Objective Function
                               │
                               ▼
                           TDS(S)
                               │
                               ▼
                            Sample
                               │
                               ▼
                           Dataset
                               │
                               ▼
                         ML Predictor
                               │
                               ▼
                           Prediction
                               │
                               ▼
                           Evaluation
```

---

# 28. Organização do projeto

A estrutura atual proposta é:

```text
csop-tradicional-learning/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   └── loader.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── engine.py
│   │   ├── global_features.py
│   │   ├── local_features.py
│   │   └── pooling.py
│   │
│   ├── subgraphs/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── networkx_generator.py
│   │
│   ├── objectives/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── tds.py
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── sample.py
│   │   └── builder.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── ...
│   │
│   └── pipeline/
│       ├── __init__.py
│       └── ...
│
├── experiments/
│
├── tests/
│   ├── test_graph.py
│   ├── test_loader.py
│   ├── test_features.py
│   ├── test_pooling.py
│   ├── test_engine.py
│   ├── test_networkx_generator.py
│   ├── test_tds.py
│   ├── test_sample.py
│   └── test_builder.py
│
├── docs/
│   └── architecture.md
│
├── requirements.txt
└── README.md
```

Essa estrutura deve ser considerada uma arquitetura evolutiva. Novos componentes podem ser adicionados conforme os experimentos demonstrarem a necessidade.

---

# 29. Primeira configuração experimental

A primeira implementação deve permanecer deliberadamente simples.

### Grafo completo

```text
- número de vértices
- número de arestas
```

### Características locais

Inicialmente:

```text
- grau dos nós
```

### Pooling

Inicialmente:

```text
- média
- máximo
- mínimo
- desvio padrão
```

### Geração de subgrafos

Inicialmente:

```text
NetworkX
generate_random_paths
```

### Função objetivo

```text
TDS
```

### Modelo

```text
LightGBM
```

O objetivo é validar o fluxo completo antes de investigar representações mais sofisticadas.

---

# 30. Integração futura com Java

A arquitetura também deve permitir que a parte de otimização seja posteriormente implementada ou mantida em Java.

Uma possível divisão é:

```text
                         JAVA
              ┌─────────────────────┐
              │ Metaheurística      │
              │                     │
              │ geração de soluções │
              │ exploração          │
              └──────────┬──────────┘
                         │
                         │ interface
                         ▼
                       PYTHON
              ┌─────────────────────┐
              │ Features            │
              │ Subgraphs           │
              │ Pooling             │
              │ ML Model            │
              │ Prediction          │
              └──────────┬──────────┘
                         │
                         │ predicted value
                         ▼
                         JAVA
```

A comunicação entre os ambientes deverá ocorrer por uma interface bem definida.

Neste momento, essa integração não será implementada. A prioridade é validar a arquitetura e o pipeline em Python.

---

# 31. Decisão arquitetural consolidada

A arquitetura foi definida com base nas seguintes decisões:

1. **Python será utilizado na primeira implementação.**
2. **O grafo completo será carregado uma única vez por instância.**
3. **Subgrafos candidatos serão gerados a partir do grafo completo.**
4. **Cada par grafo–subgrafo será tratado como uma amostra potencial do dataset.**
5. **Características globais poderão ser calculadas uma vez e reutilizadas entre os candidatos.**
6. **Características locais serão calculadas sobre os subgrafos candidatos.**
7. **Escopo (global/local) e nível (nó/aresta/grafo) serão conceitos independentes.**
8. **Pooling será responsável por transformar características de nós ou arestas em representações de tamanho fixo.**
9. **A função objetivo será independente das características e do modelo.**
10. **O `DatasetBuilder` será responsável apenas por orquestrar a construção dos samples.**
11. **O TDS será inicialmente utilizado como função objetivo de referência.**
12. **LightGBM será o primeiro modelo de regressão.**
13. **Strategy será o principal padrão de projeto.**
14. **Decorator será utilizado apenas quando houver um comportamento adicional que justifique sua aplicação.**
15. **A arquitetura permanecerá preparada para novos geradores, características, objetivos e modelos.**
16. **A construção do dataset será separada do treinamento e da predição.**
17. **A arquitetura deverá permitir uma futura integração com Java sem acoplamento direto entre as implementações.**

A principal preocupação nesta etapa é manter a arquitetura simples o suficiente para permitir experimentação, mas estruturada o suficiente para que novas hipóteses possam ser incorporadas sem reconstruir o sistema.
