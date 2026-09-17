# Arquitetura do Pipeline de Predição do Valor da Função Objetivo

## 1. Contexto

O projeto tem como objetivo investigar o uso de técnicas de Aprendizado de Máquina para a predição do valor da função objetivo associada ao problema de otimização em grafos. Nesta primeira etapa, será utilizado o **Triangle Densest Subgraph (TDS)** como função objetivo de referência.

A ideia central é representar um grafo por meio de um conjunto de características estruturais e utilizar essa representação como entrada para um modelo de regressão. A primeira modelagem utilizará o **LightGBM** para predizer o valor da função objetivo.

O trabalho ainda possui caráter exploratório. Portanto, a arquitetura deve permitir que diferentes características, estratégias de agregação, modelos de aprendizado e formas de avaliação sejam experimentados sem exigir alterações significativas nas demais partes do sistema.

A arquitetura definida deve, portanto, atender principalmente aos seguintes requisitos:

* modularidade;
* facilidade de expansão;
* baixo acoplamento entre os componentes;
* facilidade para realização de experimentos;
* separação entre extração de características, treinamento, predição e avaliação;
* possibilidade de substituição do modelo de aprendizado;
* possibilidade de inclusão de novas características do grafo;
* possibilidade de inclusão de novas estratégias de pooling;
* possibilidade de futura integração com a implementação da metaheurística em Java.

---

# 2. Pipeline geral

O pipeline inicialmente definido é composto pelas seguintes etapas:

1. cálculo das características gerais do grafo;
2. cálculo das características locais dos vértices;
3. aplicação de uma estratégia de pooling para transformar as características locais em uma representação do grafo;
4. utilização do vetor de características como entrada para o modelo LightGBM;
5. comparação da predição com o valor real da função objetivo e cálculo do erro.

De forma simplificada:

```text
Grafo
  │
  ▼
Características gerais
  │
  │  |V|, |E|, ...
  ▼
Características locais
  │
  │  grau(v), triângulos(v), ...
  ▼
Pooling
  │
  │  média, máximo, mínimo, desvio padrão, ...
  ▼
Vetor de características do grafo
  │
  ▼
LightGBM
  │
  ▼
Valor previsto
  │
  ▼
Comparação com valor real
  │
  ▼
Erro
```

A primeira implementação será propositalmente simples. Inicialmente, serão utilizadas apenas:

* número de vértices;
* número de arestas;
* grau dos vértices.

As características locais serão posteriormente agregadas por pooling para produzir uma representação de tamanho fixo do grafo.

Essa primeira versão não tem como objetivo obter imediatamente uma previsão precisa. Seu objetivo é validar o funcionamento do pipeline e estabelecer uma primeira modelagem sobre a qual novos experimentos poderão ser realizados.

---

# 3. Separação entre treinamento e predição

Embora o pipeline de predição possa ser representado como uma sequência única, o sistema precisa distinguir duas situações diferentes:

### Treinamento

Para treinar o modelo, será necessário construir um dataset a partir de diversos grafos para os quais o valor real da função objetivo seja conhecido.

```text
Vários grafos
     │
     ▼
Extração de características
     │
     ▼
Pooling
     │
     ▼
Matriz X de características
     │
     ├──────────────► Valores reais y
     │
     ▼
Dataset X, y
     │
     ▼
Treinamento
     │
     ▼
Modelo treinado
```

### Predição

Depois de treinado, o modelo poderá receber um novo grafo:

```text
Novo grafo
    │
    ▼
Extração de características
    │
    ▼
Pooling
    │
    ▼
Vetor de características
    │
    ▼
Modelo treinado
    │
    ▼
Valor previsto
```

Essa separação é importante porque **construir as características de um grafo e treinar um modelo são responsabilidades diferentes**.

---

# 4. Princípios arquiteturais

A arquitetura será guiada por quatro princípios principais.

## 4.1. Alta modularidade

Cada componente deve possuir uma responsabilidade bem definida.

Por exemplo:

* uma característica deve saber calculá-la;
* o pooling deve saber agregar valores;
* o modelo deve saber treinar e realizar predições;
* o avaliador deve saber calcular as métricas;
* o pipeline deve apenas coordenar essas operações.

Isso evita concentrar toda a lógica em uma única classe ou arquivo.

---

## 4.2. Baixo acoplamento

Os componentes não devem depender diretamente de detalhes de implementação de outros componentes.

Por exemplo, uma característica do grafo não deve depender do LightGBM.

Da mesma forma, o modelo de aprendizado não deve precisar conhecer os detalhes de como o grau dos vértices foi calculado.

A relação desejada é:

```text
Características ──────► Vetor de características
                              │
                              ▼
                           Modelo
```

e não:

```text
DegreeFeature ──────► LightGBM
```

A característica deve produzir seus dados independentemente do modelo que posteriormente os utilizará.

---

## 4.3. Facilidade de experimentação

Como o projeto possui caráter exploratório, deve ser possível testar diferentes configurações sem modificar a estrutura principal do sistema.

Por exemplo:

```text
Experimento A
|V| + |E| + média do grau

Experimento B
|V| + |E| + média + máximo do grau

Experimento C
|V| + |E| + grau + triângulos

Experimento D
|V| + |E| + outras características
```

Da mesma maneira, deve ser possível substituir o modelo:

```text
LightGBM
Random Forest
XGBoost
outro modelo de regressão
```

sem modificar as etapas de extração de características.

---

## 4.4. Separação da futura integração com Java

A arquitetura também deve considerar que a implementação da metaheurística e/ou do algoritmo de otimização poderá permanecer em Java.

Nesse cenário, Python ficará responsável principalmente pela parte de:

* extração de características;
* construção do dataset;
* treinamento;
* predição;
* avaliação do modelo.

Java poderá permanecer responsável pela parte de otimização.

A comunicação entre os dois ambientes deverá ocorrer por meio de uma interface bem definida, evitando que os componentes internos de Python dependam diretamente da implementação Java.

---

# 5. Organização inicial do projeto

A primeira organização de diretórios proposta é:

```text
tds_prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   │
│   ├── graph/
│   │   ├── graph.py
│   │   └── loader.py
│   │
│   ├── features/
│   │   ├── global_features.py
│   │   ├── local_features.py
│   │   └── pooling.py
│   │
│   ├── dataset/
│   │   └── builder.py
│   │
│   ├── model/
│   │   └── lightgbm_model.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   └── pipeline/
│       ├── feature_pipeline.py
│       ├── training_pipeline.py
│       └── prediction_pipeline.py
│
├── experiments/
│   └── first_experiment.py
│
├── tests/
│
├── requirements.txt
└── README.md
```

Essa estrutura é inicial e poderá ser modificada conforme o projeto evoluir.

---

# 6. Responsabilidade dos diretórios

## `data/`

Armazena os dados utilizados pelos experimentos.

### `data/raw/`

Dados originais, sem transformações.

### `data/processed/`

Dados preparados para utilização nos experimentos, como datasets já processados.

A separação evita misturar dados originais com resultados derivados do processamento.

---

## `src/graph/`

Responsável pela representação e carregamento dos grafos.

### `graph.py`

Deve conter a representação utilizada internamente pelo sistema.

Uma possível abstração inicial é:

```python
class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_edges(self):
        return len(self.edges)
```

### `loader.py`

Responsável por carregar grafos a partir de arquivos ou outras fontes.

Essa separação permite alterar posteriormente o formato dos dados sem modificar as etapas de extração de características.

---

# 7. Características do grafo

As características serão divididas em dois grupos:

* características globais;
* características locais.

Essa separação representa uma diferença conceitual importante.

## 7.1. Características globais

Descrevem diretamente propriedades do grafo como um todo.

Inicialmente:

```text
Número de vértices
Número de arestas
```

Uma possível abstração é:

```python
class GlobalFeature:
    def calculate(self, graph):
        raise NotImplementedError
```

Exemplos futuros:

```text
Número de vértices
Número de arestas
Densidade
Grau médio
outras características globais
```

---

# 8. Características locais

As características locais são calculadas individualmente para os vértices.

Inicialmente será utilizado o grau:

```text
grau(v)
```

Uma possível abstração é:

```python
class LocalFeature:
    def calculate(self, graph):
        raise NotImplementedError
```

A implementação inicial pode ser:

```python
class DegreeFeature(LocalFeature):

    def calculate(self, graph):
        return {
            vertex: graph.degree(vertex)
            for vertex in graph.vertices
        }
```

O resultado permanece no nível dos vértices:

```text
v1 → 3
v2 → 5
v3 → 2
v4 → 8
...
```

Essa informação ainda não representa o grafo como um todo.

Por isso, uma etapa posterior de pooling será necessária.

---

# 9. Pooling

O pooling tem como responsabilidade transformar um conjunto de características locais em uma representação de tamanho fixo do grafo.

Por exemplo:

```text
graus dos vértices

[3, 5, 2, 8, 4, ...]

        │
        ▼

   Pooling

        │
        ▼

média = 4.4
máximo = 8
mínimo = 2
desvio padrão = ...
```

A ideia é permitir diferentes estratégias de pooling.

Uma abstração possível é:

```python
class Pooling:
    def apply(self, values):
        raise NotImplementedError
```

Exemplos:

```text
MeanPooling
MaxPooling
MinPooling
StdPooling
```

Também é possível combinar várias estatísticas:

```text
grau
 │
 ├── média
 ├── máximo
 ├── mínimo
 └── desvio padrão
```

produzindo:

```text
[mean_degree,
 max_degree,
 min_degree,
 std_degree]
```

Essa etapa será particularmente importante para os experimentos, pois diferentes formas de agregação podem produzir representações diferentes para o mesmo conjunto de características locais.

---

# 10. Representação final do grafo

Depois da extração das características globais e do pooling das características locais, o grafo será representado por um vetor de tamanho fixo.

Por exemplo:

```text
[
    num_vertices,
    num_edges,
    degree_mean,
    degree_max,
    degree_min,
    degree_std
]
```

Ou, conceitualmente:

```text
Grafo
 │
 ├── |V|
 ├── |E|
 │
 └── graus
       │
       ├── média
       ├── máximo
       ├── mínimo
       └── desvio padrão
             │
             ▼
       Vetor do grafo
```

É importante manter os nomes das características internamente, em vez de trabalhar apenas com posições numéricas.

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

Isso facilita a interpretação dos experimentos e a identificação das características utilizadas.

---

# 11. Dataset Builder

O `DatasetBuilder` será responsável por transformar diversos grafos em um dataset de aprendizado supervisionado.

Para cada grafo:

```text
Grafo
  │
  ▼
Feature Pipeline
  │
  ▼
Vetor X
  │
  └────── Valor real da função objetivo y
```

Assim:

```text
G1 → X1 → y1
G2 → X2 → y2
G3 → X3 → y3
...
```

resultando em:

```text
X = características dos grafos

y = valores reais da função objetivo
```

Uma possível interface:

```python
class DatasetBuilder:

    def __init__(self, feature_pipeline):
        self.feature_pipeline = feature_pipeline

    def build(self, graphs):
        ...
```

O `DatasetBuilder` não deve ser responsável por treinar o modelo.

Sua responsabilidade termina na construção do dataset.

---

# 12. Modelo de aprendizado

O modelo será inicialmente implementado utilizando LightGBM.

Entretanto, a arquitetura não deve fazer o restante do sistema depender diretamente do LightGBM.

Uma abstração possível:

```python
class TDSPredictor:

    def train(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
```

A implementação inicial:

```python
class LightGBMTDSPredictor(TDSPredictor):

    def train(self, X, y):
        ...

    def predict(self, X):
        ...
```

Dessa forma, futuramente outros modelos poderão implementar a mesma interface:

```text
TDSPredictor
    │
    ├── LightGBMTDSPredictor
    ├── RandomForestPredictor
    ├── XGBoostPredictor
    └── ...
```

O restante do pipeline não precisa saber qual modelo concreto está sendo utilizado.

---

# 13. Avaliação

A avaliação será uma etapa independente do modelo.

Sua responsabilidade será comparar:

```text
valor real
     ×
valor previsto
```

e calcular o erro.

Uma abstração inicial:

```python
class Evaluator:

    def calculate_error(self, real, predicted):
        return abs(real - predicted)
```

Posteriormente, poderão ser incorporadas outras métricas, como:

```text
MAE
RMSE
MAPE
R²
```

A escolha das métricas poderá ser definida conforme os experimentos e os objetivos da pesquisa.

---

# 14. Padrões de projeto

A arquitetura poderá utilizar diferentes padrões de projeto, mas cada um deve ser aplicado para resolver uma necessidade concreta.

Os principais padrões considerados são:

* Strategy;
* Decorator;
* Factory;
* Facade/Pipeline.

A intenção não é utilizar padrões apenas por formalidade. Eles devem contribuir para a modularidade e extensibilidade do sistema.

---

# 15. Strategy

O **Strategy** será o principal padrão de projeto da arquitetura.

Ele é adequado porque o projeto possui diversos componentes que podem ser substituídos por diferentes estratégias.

### Características

```text
LocalFeature
    │
    ├── DegreeFeature
    ├── TriangleFeature
    ├── CentralityFeature
    └── ...
```

### Pooling

```text
Pooling
    │
    ├── MeanPooling
    ├── MaxPooling
    ├── MinPooling
    └── StdPooling
```

### Modelos

```text
Predictor
    │
    ├── LightGBM
    ├── Random Forest
    └── ...
```

A principal vantagem é permitir substituir uma estratégia sem modificar o código que a utiliza.

Por exemplo:

```text
FeaturePipeline
       │
       ├── DegreeFeature
       │
       └── MeanPooling
```

pode ser alterado para:

```text
FeaturePipeline
       │
       ├── DegreeFeature
       │
       └── MaxPooling
```

sem modificar o restante do pipeline.

---

# 16. Decorator

O padrão **Decorator** será considerado para comportamentos que possam ser adicionados às características ou componentes sem alterar suas implementações originais.

Ele não será utilizado para representar novas características.

Por exemplo:

```text
DegreeFeature
     │
     ▼
NormalizationDecorator
```

ou:

```text
TriangleFeature
     │
     ▼
CacheDecorator
```

Possíveis utilizações futuras incluem:

* normalização;
* cache de resultados;
* registro de execução;
* medição de tempo;
* outras transformações ou comportamentos auxiliares.

A distinção conceitual será:

> **Strategy define qual estratégia será utilizada, enquanto Decorator adiciona ou modifica um comportamento dessa estratégia.**

Assim, uma nova característica, como `TriangleFeature`, deve ser tratada como uma nova Strategy e não como um Decorator de `DegreeFeature`.

---

# 17. Factory

O padrão **Factory** poderá ser utilizado para centralizar a criação dos componentes.

Isso se torna particularmente útil à medida que o número de características, estratégias de pooling e modelos aumentar.

Por exemplo, uma configuração poderia indicar:

```yaml
features:
  global:
    - num_vertices
    - num_edges

  local:
    - degree

pooling:
  - mean
  - max

model:
  type: lightgbm
```

Factories poderiam interpretar essa configuração e construir os componentes necessários.

Conceitualmente:

```text
Configuração
     │
     ▼
FeatureFactory
     │
     ├── NumVertices
     ├── NumEdges
     └── DegreeFeature

PoolingFactory
     │
     ├── MeanPooling
     └── MaxPooling

ModelFactory
     │
     └── LightGBM
```

Isso permitirá futuramente modificar configurações de experimentos sem alterar a lógica principal do sistema.

---

# 18. Facade / Pipeline

O pipeline funcionará como uma camada de orquestração.

Sua função será coordenar os componentes, e não implementar diretamente seus cálculos.

Por exemplo:

```text
FeaturePipeline
     │
     ├── GlobalFeatureExtractor
     │
     ├── LocalFeatureExtractor
     │
     └── FeaturePooler
```

O pipeline pode ser responsável por executar:

```text
1. extrair características globais;
2. extrair características locais;
3. aplicar pooling;
4. combinar os resultados;
5. retornar o vetor de características.
```

Ele não precisa saber como cada característica ou pooling é implementado.

Essa separação mantém o pipeline simples e facilita a substituição dos componentes internos.

---

# 19. Arquitetura combinando os padrões

A arquitetura conceitual completa pode ser representada da seguinte forma:

```text
                         CONFIGURAÇÃO
                              │
                              ▼
                           FACTORY
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          Features          Pooling          Model
              │               │               │
           Strategy         Strategy        Strategy
              │
          Decorators
              │
              ▼
       Feature Pipeline
              │
              ▼
      Vetor de características
              │
              ▼
        Dataset Builder
              │
              ▼
       Training Pipeline
              │
              ▼
       Modelo LightGBM
              │
              ▼
     Prediction Pipeline
              │
              ▼
         Avaliação
```

Essa arquitetura permite que os componentes sejam desenvolvidos e testados de forma independente.

---

# 20. Primeira implementação

Apesar da arquitetura ser preparada para expansão, a primeira versão deve ser deliberadamente pequena.

A configuração inicial será:

### Características globais

```text
- número de vértices
- número de arestas
```

### Característica local

```text
- grau dos vértices
```

### Pooling

Inicialmente, uma ou algumas estatísticas simples, por exemplo:

```text
- média
- máximo
- mínimo
- desvio padrão
```

### Modelo

```text
LightGBM
```

### Saída

```text
Valor previsto da função objetivo do TDS
```

### Avaliação

```text
Valor real
Valor previsto
Erro
```

O objetivo dessa primeira implementação é verificar se todo o fluxo funciona:

```text
Grafo
  ↓
Características
  ↓
Pooling
  ↓
Vetor
  ↓
LightGBM
  ↓
Predição
  ↓
Erro
```

Não se pretende, nessa etapa, definir a representação final mais adequada ou obter imediatamente bons resultados de predição.

---

# 21. Integração futura com Java

A arquitetura deverá manter uma fronteira clara entre a parte de otimização e a parte de aprendizado.

Uma possível organização futura será:

```text
                         JAVA
              ┌──────────────────────┐
              │ Metaheurística       │
              │                      │
              │ geração de soluções  │
              │ avaliação/exploração │
              └──────────┬───────────┘
                         │
                         │ interface
                         ▼
                       PYTHON
              ┌──────────────────────┐
              │ Feature Pipeline     │
              │          ↓           │
              │ Pooling              │
              │          ↓           │
              │ Modelo ML            │
              │          ↓           │
              │ Predição             │
              └──────────┬───────────┘
                         │
                         │ valor previsto
                         ▼
                         JAVA
```

A comunicação poderá posteriormente utilizar uma interface baseada em dados simples, como estruturas serializadas ou JSON.

Por exemplo, uma representação poderia ser:

```json
{
    "num_vertices": 100,
    "num_edges": 250,
    "degree_mean": 5.0,
    "degree_max": 21,
    "degree_min": 1
}
```

e a resposta:

```json
{
    "predicted_tds": 31.7
}
```

Essa comunicação, entretanto, não será implementada na primeira versão. Neste momento, a prioridade é validar a arquitetura e o pipeline em Python.

---

# 22. Decisão arquitetural consolidada

A arquitetura escolhida pode ser resumida nos seguintes pontos:

1. **Python será utilizado na primeira implementação**, concentrando a extração de características, construção do dataset, treinamento, predição e avaliação.

2. **O sistema será modular**, separando grafo, características, pooling, dataset, modelo, avaliação e pipelines.

3. **Strategy será o principal padrão de projeto**, permitindo trocar características, estratégias de pooling e modelos.

4. **Decorator será utilizado de maneira pontual**, principalmente para adicionar comportamentos às características ou estratégias, como cache, normalização ou monitoramento, sem confundi-lo com a representação de novas características.

5. **Factory poderá ser utilizada para criação dos componentes**, principalmente quando o número de configurações experimentais aumentar.

6. **O pipeline funcionará como camada de orquestração**, mantendo a lógica de execução separada das implementações específicas.

7. **A construção do dataset será separada do treinamento**, pois são etapas conceitualmente distintas.

8. **O modelo será isolado da representação do grafo**, recebendo apenas os vetores de características.

9. **A avaliação será independente do modelo**, permitindo testar diferentes métricas posteriormente.

10. **A arquitetura será preparada para futura integração com Java**, mantendo uma interface bem definida entre a parte de otimização e a parte de aprendizado.

---

# 23. Arquitetura inicial consolidada

A primeira versão do projeto pode ser resumida como:

```text
tds_prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   │
│   ├── graph/
│   │   ├── graph.py
│   │   └── loader.py
│   │
│   ├── features/
│   │   ├── global_features.py
│   │   ├── local_features.py
│   │   └── pooling.py
│   │
│   ├── dataset/
│   │   └── builder.py
│   │
│   ├── model/
│   │   └── lightgbm_model.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   └── pipeline/
│       ├── feature_pipeline.py
│       ├── training_pipeline.py
│       └── prediction_pipeline.py
│
├── experiments/
│   └── first_experiment.py
│
├── tests/
│
├── requirements.txt
└── README.md
```

Essa estrutura deve ser considerada uma **arquitetura inicial**, e não uma especificação definitiva. Conforme os experimentos revelarem novas necessidades, componentes poderão ser reorganizados ou abstrações poderão ser introduzidas.

A principal preocupação neste estágio é garantir que novas hipóteses possam ser incorporadas ao sistema de maneira incremental, preservando a separação entre **representação do grafo, extração de características, agregação, aprendizado e avaliação**.
