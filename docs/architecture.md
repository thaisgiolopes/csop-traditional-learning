# CSOP Traditional Learning

Investigação de técnicas de aprendizado de máquina para predição do valor da função objetivo em problemas de otimização em grafos.

## Sobre o projeto

Este projeto investiga a utilização de características estruturais de grafos para representar instâncias do **Connected Subgraph Optimization Problem (CSOP)** e predizer o valor de sua função objetivo por meio de modelos de aprendizado de máquina.

A primeira etapa utiliza o **Triangle Densest Subgraph (TDS)** como função objetivo de referência e emprega características estruturais tradicionais do grafo, como número de vértices, número de arestas e características locais dos vértices.

O pipeline inicial é:

```text
Grafo
  ↓
Características gerais
  ↓
Características locais
  ↓
Pooling
  ↓
Vetor de características
  ↓
Modelo de regressão
  ↓
Predição do valor da função objetivo
  ↓
Avaliação do erro
```

## Objetivo inicial

A primeira implementação tem como objetivo construir e validar um pipeline básico para:

1. extrair características gerais do grafo;
2. extrair características locais dos vértices;
3. agregar as características locais por meio de pooling;
4. construir uma representação vetorial do grafo;
5. treinar um modelo de regressão;
6. predizer o valor da função objetivo do TDS;
7. comparar a predição com o valor real.

Nesta etapa, o foco está na construção e validação do pipeline, e não na obtenção imediata de alta precisão de predição.

## Arquitetura

O projeto é organizado de forma modular, separando:

* representação e carregamento dos grafos;
* extração de características;
* pooling;
* construção do dataset;
* modelos de aprendizado;
* avaliação;
* pipelines de treinamento e predição.

A arquitetura utiliza conceitos de **Strategy**, **Decorator**, **Factory** e **Facade/Pipeline**, de acordo com as necessidades de cada componente.

A documentação da arquitetura está disponível em [`docs/architecture.md`](docs/architecture.md).

## Estrutura

```text
src/
├── graph/
├── features/
├── dataset/
├── model/
├── evaluation/
└── pipeline/

experiments/
tests/
data/
├── raw/
└── processed/

docs/
```

## Tecnologias

* Python
* LightGBM
* Bibliotecas para manipulação e análise de grafos
* Bibliotecas para construção e avaliação de modelos de aprendizado de máquina

## Status

🚧 Em desenvolvimento.

O projeto está atualmente na etapa de definição e implementação do primeiro pipeline experimental.
