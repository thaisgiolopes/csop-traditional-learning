# Planejamento da apresentação do projeto

**Duração prevista:** 25 minutos, com possibilidade de expansão até 30 minutos  
**Público:** orientação do projeto e docente com experiência em aprendizado de máquina  
**Experimento de referência:** `EXP_20261111_307605b45817`

## Objetivo da apresentação

Apresentar o problema investigado, a formulação da tarefa supervisionada, o pipeline experimental e os resultados atuais. A apresentação também deve explicitar as escolhas e limitações do protocolo de aprendizado de máquina, para orientar a discussão sobre as próximas etapas.

## Estrutura e duração

| Seção | Tempo |
|---|---:|
| 1. Problema e motivação | 2 min |
| 2. Formulação do CSOP e função objetivo | 3 min |
| 3. Dados e geração dos candidatos | 2 min |
| 4. Representação e atributos | 3 min |
| 5. Pipeline experimental | 2 min |
| 6. Protocolo de avaliação e caracterização dos splits | 5 min |
| 7. Modelo e hiperparâmetros | 3 min |
| 8. Resultados | 3 min |
| 9. Limitações, questões de ML e próximos passos | 2 min |
| **Total** | **25 min** |

Para uma versão de 30 minutos, reservar cinco minutos adicionais para discussão das perguntas metodológicas da seção 9.

---

## 1. Problema e motivação — 2 min

### Conteúdo

- O projeto investiga a previsão de valores de função objetivo para candidatos do Problema de Otimização de Subgrafos Conexos (CSOP).
- Cada candidato é um subgrafo de um grafo de entrada.
- A motivação de longo prazo é investigar se estimativas podem ajudar a priorizar candidatos e reduzir avaliações diretas da função objetivo.
- O escopo atual é construir e avaliar um preditor. A redução de custo dentro de um algoritmo de otimização ainda não foi demonstrada.

### Recurso visual sugerido

Diagrama simples: grafo de entrada → conjunto de subgrafos candidatos → atributos e valores da função objetivo.

## 2. Formulação do CSOP e função objetivo — 3 min

### Conteúdo

- Apresentar o problema como seleção ou busca entre subgrafos candidatos que atendam às condições definidas, incluindo conectividade.
- Introduzir a função objetivo utilizada: Triangle Densest Subgraph (TDS).
- Definição:

  \[
  TDS(S) = \frac{T(S)}{|V(S)|}
  \]

  onde `T(S)` é o número de triângulos em `S` e `|V(S)|` é o número de vértices.

- A tarefa de ML é regressão supervisionada: os atributos do candidato são as entradas e a TDS calculada diretamente é o alvo contínuo.
- A implementação conta triângulos no subgrafo e divide pelo número de vértices.

### Ponto de precisão

O modelo prediz o valor de TDS; ele não substitui o algoritmo de busca nem, nesta etapa, decide sozinho qual candidato deve ser selecionado.

## 3. Dados e geração dos candidatos — 2 min

### Conteúdo do experimento de referência

- Grafo de origem: Amazon.
- Tamanho do grafo: 334.863 vértices e 925.872 arestas.
- Geração de subgrafos conexos com tamanho exato por expansão conectada.
- Tamanhos avaliados: 10 a 500 vértices, em incrementos de 10 (50 tamanhos).
- Candidatos por tamanho: 100.
- Total: 5.000 amostras candidatas.
- Semente registrada: 42; limite de tentativas de geração: 5.000.

### Ponto de interpretação

As 5.000 amostras são subgrafos candidatos derivados de um único grafo de origem. O experimento, portanto, não mede generalização entre grafos de origem.

## 4. Representação e atributos — 3 min

### Atributos usados

| Atributo | Interpretação |
|---|---|
| `num_vertices` | Número de vértices do subgrafo candidato |
| `num_edges` | Número de arestas do subgrafo candidato |
| `degree_mean` | Média dos graus dos vértices do candidato |
| `degree_max` | Maior grau no candidato |
| `degree_min` | Menor grau no candidato |
| `degree_std` | Desvio-padrão dos graus no candidato |

As estatísticas de grau são obtidas calculando os graus locais e agregando-os por média, máximo, mínimo e desvio-padrão.

### Pontos para contextualizar

- As contagens do grafo de origem ficam nos metadados; não são atributos do modelo.
- O tamanho do candidato é a própria feature `num_vertices` e também define a ordem dos folds; não é repetido como `subgraph_size`.
- A contribuição dos atributos ainda não foi estudada por importância de variáveis ou ablação.

## 5. Pipeline experimental — 2 min

Apresentar o fluxo de processamento:

1. Carregamento do grafo Amazon.
2. Geração de candidatos conexos nos tamanhos especificados.
3. Extração de atributos globais e locais.
4. Agregação das estatísticas de grau.
5. Cálculo direto da TDS para compor o alvo.
6. Construção e persistência do conjunto tabular.
7. Separação em desenvolvimento e teste.
8. Treinamento, validação cruzada, ajuste final e cálculo das métricas.

### Recurso visual sugerido

Fluxograma horizontal com uma etapa por bloco; destacar que a TDS calculada é o rótulo supervisionado.

## 6. Protocolo de avaliação e splits — 5 min

### Separação inicial

- 20% do conjunto completo foram reservados para o teste final: 1.000 amostras.
- O holdout foi estratificado por tamanho: 20 candidatos de cada um dos 50 tamanhos.
- Os 4.000 candidatos restantes formam o conjunto de desenvolvimento.
- A semente de split é 42.

### Validação cruzada progressiva por tamanho

Foi configurado `TimeSeriesSplit` com cinco folds, aplicado à lista ordenada de tamanhos no conjunto de desenvolvimento. Cada bloco de validação contém oito tamanhos (640 amostras); o treino acumula tamanhos à medida que os folds avançam.

| Fold | Tamanhos de treino | N de treino | Tamanhos de validação | N de validação |
|---|---:|---:|---:|---:|
| 1 | 10–100 | 800 | 110–180 | 640 |
| 2 | 10–180 | 1.440 | 190–260 | 640 |
| 3 | 10–260 | 2.080 | 270–340 | 640 |
| 4 | 10–340 | 2.720 | 350–420 | 640 |
| 5 | 10–420 | 3.360 | 430–500 | 640 |

### Terminologia e interpretação

- **Número de folds:** cinco.
- **Janela de treino:** expansiva; inclui os tamanhos anteriores e cresce a cada fold.
- **Janela de validação:** bloco de oito tamanhos consecutivos neste experimento.
- **Tamanho fixo de treino:** não foi definido; o número de exemplos de treino cresce de 800 para 3.360.
- **Série temporal:** os dados não possuem timestamps e não formam uma série temporal literal. A ordem usada é a dos tamanhos de subgrafo. Recomenda-se descrever o método como validação progressiva ou para frente por tamanho, inspirada em estratégias de séries temporais.

### Distinção fundamental entre CV e teste final

A validação cruzada mede desempenho em blocos de tamanhos maiores que os vistos no treino de cada fold. Já o teste final contém exemplos de todos os tamanhos, inclusive tamanhos presentes no conjunto de desenvolvimento. Assim:

- os resultados dos folds são os mais diretamente relacionados à generalização para tamanhos maiores;
- as métricas do teste final descrevem desempenho em novos candidatos amostrados dentro da distribuição de tamanhos representada no desenvolvimento;
- o R² do teste final não deve ser apresentado como evidência isolada de extrapolação para tamanhos inéditos.

### Recurso visual sugerido

Uma linha de tamanhos de 10 a 500, com cores distintas para treino e validação em cada fold. Mostrar o teste final em um painel separado como amostragem estratificada nos 50 tamanhos.

## 7. Modelo e hiperparâmetros — 3 min

### Modelo

- Regressor LightGBM, por meio de `LGBMRegressor`.
- Recebe a tabela de atributos e prediz um valor contínuo de TDS.
- Um modelo é ajustado em cada fold; depois, o modelo final é reajustado em todo o conjunto de desenvolvimento e avaliado no teste.

### Justificativa atual

LightGBM foi adotado como primeiro modelo por ser apropriado para dados tabulares, captar relações não lineares e interações entre atributos, e exigir pouco pré-processamento. A escolha ainda é exploratória: não houve comparação sistemática com outros modelos nem busca de hiperparâmetros.

### Hiperparâmetros registrados

| Parâmetro | Valor |
|---|---:|
| `n_estimators` | 100 |
| `learning_rate` | 0,05 |
| `num_leaves` | 31 |
| `min_child_samples` | 20 |
| `n_jobs` | 1 |
| `random_state` | 42 |
| `verbosity` | -1 |

### Questão metodológica para destacar

Árvores de decisão particionam o espaço de atributos em regiões e, em geral, não extrapolam tendências numéricas além dos valores vistos no treino da mesma forma que modelos lineares. Como a validação progressiva usa valores de `num_vertices` maiores que os tamanhos de treino, esse comportamento deve ser analisado por fold e comparado com alternativas.

## 8. Resultados — 3 min

### Validação cruzada

| Fold | MAE | RMSE | R² |
|---|---:|---:|---:|
| 1 | 0,111 | 0,194 | 0,835 |
| 2 | 0,073 | 0,105 | 0,940 |
| 3 | 0,063 | 0,096 | 0,938 |
| 4 | 0,056 | 0,079 | 0,934 |
| 5 | 0,052 | 0,070 | 0,934 |
| **Média** | **0,071** | **0,109** | **0,916** |

Desvios-padrão entre folds: MAE = 0,021; RMSE = 0,044; R² = 0,041.

### Teste final estratificado

| Métrica | Resultado |
|---|---:|
| MAE | 0,061 |
| RMSE | 0,098 |
| R² | 0,942 |
| Número de amostras | 1.000 |

### Interpretação das métricas

- **MAE:** erro absoluto médio, na escala da TDS.
- **RMSE:** erro quadrático médio em raiz, também na escala da TDS e mais sensível a erros grandes.
- **R²:** desempenho relativo a uma previsão de referência baseada na média; não é acurácia nem significa “percentual de previsões corretas”.

Os valores dos folds diferem. A melhora do MAE ao longo dos folds pode estar relacionada ao aumento do conjunto de treino, mas também há mudança nos intervalos de tamanho e nas distribuições avaliadas. Não atribuir a variação a uma única causa sem análise adicional.

## 9. Limitações, questões de ML e próximos passos — 2 min, mais discussão opcional

### Limitações atuais

- Um único grafo de origem.
- Candidatos derivados do mesmo grafo podem compartilhar estrutura; a independência estatística entre amostras não foi estabelecida.
- Um único método de geração e uma configuração de semente no experimento apresentado.
- Nenhum baseline ou comparação sistemática entre modelos.
- Nenhuma busca de hiperparâmetros.
- Sem análise de importância/ablação dos atributos.
- Ainda não foi medida a utilidade das previsões dentro de um algoritmo de otimização, por exemplo, em qualidade de ranking, economia de avaliações ou qualidade da solução final.

### Perguntas para orientar a contribuição da professora

1. A pergunta principal deve ser generalização para tamanhos maiores, para novos candidatos dos mesmos tamanhos ou para grafos de origem diferentes?
2. A validação progressiva por tamanho é adequada para essa pergunta? O teste final deveria também reservar uma faixa inteira de tamanhos?
3. Como lidar com candidatos que podem compartilhar vértices ou arestas? Faz sentido avaliar splits por grupos ou por grafo de origem?
4. Quais baselines são necessários: média, regressão linear, Random Forest, outros modelos de boosting?
5. Como avaliar a contribuição de `num_vertices` e das estatísticas de grau por ablação e importância de atributos?
6. Quais métricas refletem melhor o uso futuro do modelo na otimização, além de MAE, RMSE e R²?
7. Como planejar experimentos com múltiplas sementes e grafos para medir robustez?

### Próximos passos possíveis

1. Definir explicitamente a população-alvo e o tipo de generalização desejado.
2. Incluir baselines simples e comparar modelos sob o mesmo protocolo.
3. Avaliar métricas por tamanho e inspecionar erros e resíduos.
4. Fazer análise de atributos e ablação.
5. Repetir com múltiplas sementes e grafos.
6. Avaliar a utilidade do modelo numa etapa concreta de busca do CSOP.

## Referências aos artefatos do experimento

- [Configuração do experimento](../experiments/runs/EXP_20261111_307605b45817/config/experiment.json)
- [Métricas](../experiments/runs/EXP_20261111_307605b45817/evaluation/metrics.json)
- [Metadados dos splits](../experiments/runs/EXP_20261111_307605b45817/datasets/split_metadata.json)
- [Metadados do conjunto de dados](../experiments/runs/EXP_20261111_307605b45817/datasets/dataset.metadata.json)
