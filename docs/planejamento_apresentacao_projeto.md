# Planejamento da apresentação do projeto

**Duração prevista:** 25–30 minutos  
**Público:** orientação do projeto e docente com experiência em aprendizado de máquina  
**Experimentos de referência:** execuções de Amazon, Patents e transferência Amazon → Patents documentadas na seção 8.

## Objetivo da apresentação

Apresentar o problema investigado, a formulação da tarefa supervisionada, o pipeline experimental e os resultados atuais. A apresentação também deve explicitar as escolhas e limitações do protocolo de aprendizado de máquina, para orientar a discussão sobre as próximas etapas.

## Estrutura e duração

| Seção | Tempo |
|---|---:|
| 1. Problema, motivação e objetivo de ML | 2 min |
| 2. Formulação do CSOP e função objetivo | 3 min |
| 3. Dados e geração de candidatos | 2 min |
| 4. Pipeline e construção do conjunto tabular | 2 min |
| 5. Atributos: definição e justificativa | 3 min |
| 6. Modelo, velocidade e hiperparâmetros | 4 min |
| 7. Desenho experimental e caracterização dos splits | 5 min |
| 8. Experimentos representativos | 3 min |
| 9. Resultados e interpretações | 5 min |
| 10. Questões metodológicas e próximos passos | 3 min |
| **Total** | **30 min** |

Para uma versão de 20 minutos, encurtar a descrição do grafo e dos hiperparâmetros e apresentar somente as métricas médias e as diferenças principais entre protocolos. Para 30 minutos, reservar tempo para discutir as perguntas de ML da seção 10.

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

### Conteúdo dos experimentos representativos

- Grafos de origem: Amazon e Patents.
- Amazon: 334.863 vértices e 925.872 arestas.
- Geração de subgrafos conexos com tamanho exato por expansão conectada.
- Tamanhos avaliados: 10 a 500 vértices, em incrementos de 10 (50 tamanhos).
- Candidatos por tamanho: 100.
- Total: 5.000 amostras candidatas.
- Semente registrada: 42; limite de tentativas de geração: 5.000.

### Ponto de interpretação

As 5.000 amostras são subgrafos candidatos derivados de um único grafo de origem. O experimento, portanto, não mede generalização entre grafos de origem.

## 4. Representação e atributos — 3 min

### Atributos usados e justificativa

| Atributo | Interpretação |
|---|---|
| `num_vertices` | Número de vértices do grafo de entrada; contextualiza a escala do grafo |
| `num_edges` | Número de arestas do grafo de entrada; contextualiza a densidade/escala da rede |
| `subgraph_size` | Número de vértices do candidato; informa diretamente o tamanho ao modelo |
| `degree_mean` | Média dos graus dos vértices do candidato |
| `degree_max` | Maior grau no candidato |
| `degree_min` | Menor grau no candidato |
| `degree_std` | Desvio-padrão dos graus no candidato |

As estatísticas de grau são obtidas calculando os graus locais e agregando-os por média, máximo, mínimo e desvio-padrão.

### Razão para incluir cada grupo de atributos

- `num_vertices` e `num_edges`: contextualizam o candidato em relação ao grafo completo. São relevantes quando se comparam grafos de escalas diferentes, como Amazon e Patents. Em um experimento feito em apenas um grafo, são constantes e não ajudam a distinguir candidatos; essa redundância deve ser considerada na análise.
- `subgraph_size`: tamanho do candidato. A função objetivo divide pelo número de vértices, e a estrutura observada pode mudar com o tamanho; fornecer essa variável ajuda o modelo a condicionar a previsão ao tamanho.
- `degree_mean`: resumo da conectividade média local; é uma estatística simples e barata de calcular.
- `degree_max` e `degree_min`: capturam extremos que a média pode esconder, por exemplo, a presença de hubs ou vértices pouco conectados.
- `degree_std`: mede heterogeneidade da conectividade dentro do candidato.
- Em conjunto, as estatísticas de grau oferecem uma representação tabular compacta de estrutura local. São um primeiro conjunto interpretável, não uma representação completa de toda a topologia.

### Pontos para contextualizar

- O vetor contém sete colunas numéricas. `num_vertices` e `num_edges` referem-se ao grafo completo; `subgraph_size` refere-se ao candidato.
- `subgraph_size` é também a chave usada para ordenar os folds do experimento `time_series_size`.
- A contribuição dos atributos ainda não foi estudada por importância de variáveis ou ablação.

## 5. Pipeline experimental — 2 min

Apresentar o fluxo de processamento:

1. Carregamento do grafo de origem (Amazon ou Patents).
2. Geração de candidatos conexos nos tamanhos especificados.
3. Extração de atributos globais e locais.
4. Agregação das estatísticas de grau.
5. Cálculo direto da TDS para compor o alvo.
6. Construção e persistência do conjunto tabular.
7. Separação em desenvolvimento e teste.
8. Treinamento e validação conforme o protocolo escolhido.
9. Ajuste do modelo final nos dados permitidos pelo protocolo e avaliação no teste reservado.
10. Persistência de configuração, metadados de split, previsões e métricas para rastreabilidade.

### Recurso visual sugerido

Fluxograma horizontal com uma etapa por bloco; destacar que a TDS calculada é o rótulo supervisionado.

## 6. Protocolo de avaliação e splits — 5 min

### Separação inicial

- 20% do conjunto completo foram reservados para o teste final: 1.000 amostras.
- O holdout foi estratificado por tamanho: 20 candidatos de cada um dos 50 tamanhos.
- Os 4.000 candidatos restantes formam o conjunto de desenvolvimento.
- A semente de split é 42.

### Split aleatório em três partições

Nos experimentos `random`, o protocolo é treino/validação/teste, aproximadamente 60%/20%/20% do total. O teste externo usa 20%. Dos 80% restantes, 25% são destinados à validação (20% do total) e 75% ao treino (60% do total). Essa avaliação mede desempenho em candidatos aleatórios, com os tamanhos representados nas partições; não mede extrapolação para tamanhos inteiramente inéditos.

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

O `TimeSeriesSplit` do projeto cria folds de treino e validação em ordem crescente de tamanho. A opção `test_ratio=0.2` reserva adicionalmente o conjunto chamado `test` por amostragem estratificada em todos os tamanhos. Assim, há treino, validação progressiva e teste final, mas o teste final não é uma faixa cronologicamente posterior de tamanhos. A apresentação deve deixar essa distinção explícita.

### Recurso visual sugerido

Uma linha de tamanhos de 10 a 500, com cores distintas para treino e validação em cada fold. Mostrar o teste final em um painel separado como amostragem estratificada nos 50 tamanhos.

## 7. Modelo, foco tabular, velocidade e hiperparâmetros — 4 min

### Modelo

- Regressor LightGBM, por meio de `LGBMRegressor`.
- Recebe a tabela de atributos e prediz um valor contínuo de TDS.
- Um modelo é ajustado em cada fold; depois, o modelo final é reajustado em todo o conjunto de desenvolvimento e avaliado no teste.

### Justificativa atual

O projeto pretende investigar primeiro modelos tabulares, que recebem estatísticas compactas calculadas para cada candidato. LightGBM foi adotado como primeiro modelo por ser apropriado para dados tabulares, captar relações não lineares e interações entre atributos, e exigir pouco pré-processamento. A escolha ainda é exploratória: não houve comparação sistemática com outros modelos nem busca de hiperparâmetros.

Não começamos com uma GNN porque a pergunta experimental atual é se atributos tabulares simples conseguem prever a TDS com custo baixo, e não aprender diretamente uma representação completa do grafo. Uma GNN é uma direção possível de comparação futura, especialmente se as estatísticas agregadas perderem informação topológica importante. Essa escolha é de escopo, não uma afirmação de que GNNs sejam inadequadas.

O custo também faz parte do objetivo de aplicação: se a extração de atributos e a inferência forem tão lentas quanto calcular diretamente a TDS, o modelo pode não trazer benefício no processo de busca. Ainda não há medição comparativa de tempo nos experimentos apresentados. Devem ser medidos separadamente o custo de geração/extração, o custo de calcular TDS, o tempo de treinamento e a latência de previsão, por candidato e em lote.

### Hiperparâmetros registrados

| Parâmetro | Valor |
|---|---:|
| `n_estimators` | 100 |
| `learning_rate` | 0,05 |
| `num_leaves` | 31 |
| `min_child_samples` | 20 |
| `n_jobs` | 1 (8 no graph transfer) |
| `random_state` | 42 |
| `verbosity` | -1 |

### Interpretação dos hiperparâmetros

- `n_estimators=100`: número de árvores/etapas de boosting. Define parte do custo e da capacidade do conjunto; valor inicial, não selecionado por busca.
- `learning_rate=0.05`: controla a contribuição de cada árvore. Uma taxa menor costuma exigir mais árvores; o valor é um compromisso inicial entre aprendizado gradual e custo.
- `num_leaves=31`: controla a capacidade das árvores. Mais folhas podem capturar relações detalhadas, mas também aumentam complexidade, custo e risco de sobreajuste.
- `min_child_samples=20`: exige um mínimo de amostras em cada folha e regulariza regras baseadas em poucos candidatos.
- `n_jobs=1` nos experimentos locais e `n_jobs=8` no graph transfer: define o paralelismo de treino. Como a configuração não foi calibrada por benchmark de latência, apresentar essa diferença e propor comparação controlada do tempo com o mesmo hardware e número de threads.
- `random_state=42`: fixa a semente para facilitar reprodutibilidade.
- `verbosity=-1`: reduz mensagens de log e não altera a qualidade do modelo.

Os valores devem ser apresentados como configuração de baseline. Não foram afinados por busca sistemática nem escolhidos após comparação de grades de parâmetros.

### Questão metodológica para destacar

Árvores de decisão particionam o espaço de atributos em regiões e, em geral, não extrapolam tendências numéricas além dos valores vistos no treino da mesma forma que modelos lineares. Como a validação progressiva usa valores de `subgraph_size` maiores que os tamanhos de treino, esse comportamento deve ser analisado por fold e comparado com alternativas.

## 8. Experimentos representativos — 3 min

Apresentar cinco execuções já registradas no repositório, cobrindo os principais regimes de avaliação.

| Regime | Grafo(s) | Execução | Uso na apresentação |
|---|---|---|---|
| Random, treino/validação/teste | Amazon | `EXP_20261021_88cae057a82d` | Referência de desempenho com divisão aleatória |
| Random, treino/validação/teste | Patents | `EXP_20261112_70f000f9ad58` | Contraste entre grafos sob o mesmo regime geral |
| Progressivo por tamanho, CV + teste | Amazon | `EXP_20261111_307605b45817` | Generalização para blocos de tamanhos maiores |
| Progressivo por tamanho, CV + teste | Patents | `EXP_20261113_8990d8ddf6e9` | Repetição da pergunta de generalização no segundo grafo |
| Transferência entre grafos | Amazon → Patents | `EXP_20261104_183446919fe2` | Teste de mudança de grafo/domínio |

### Interpretação dos regimes

- **Random:** divide candidatos aleatoriamente em treino, validação e teste. É uma referência de desempenho em candidatos novos dentro da mistura de tamanhos amostrada; não mede extrapolação para tamanhos inéditos.
- **Progressivo por tamanho:** os folds de treino e validação respeitam a ordem dos tamanhos e validam em blocos maiores que o treino de cada fold. O teste final é amostrado estratificadamente em todos os tamanhos, então não é uma faixa futura isolada.
- **Graph transfer:** treina em Amazon e testa em Patents. Os dois conjuntos têm candidatos nos tamanhos 10–500, 100 por tamanho. A avaliação externa contém 5.000 candidatos Patents.

Não comparar diretamente MAE/RMSE de grafos diferentes sem considerar escala e distribuição de TDS. R² também depende da variância do alvo. Apresentar distribuição do alvo e baselines por grafo para contextualizar.

## 9. Resultados e interpretações — 5 min

### Resumo dos resultados principais

| Regime | Grafo / direção | MAE | RMSE | R² | N de avaliação |
|---|---|---:|---:|---:|---:|
| Random, teste | Amazon | 0,062 | 0,096 | 0,940 | 1.000 |
| Random, teste | Patents | 0,025 | 0,037 | 0,816 | 1.000 |
| CV progressiva, média dos folds | Amazon | 0,071 | 0,109 | 0,916 | 3.200 previsões OOF |
| Teste estratificado | Amazon | 0,061 | 0,098 | 0,942 | 1.000 |
| CV progressiva, média dos folds | Patents | 0,029 | 0,043 | 0,679 | 3.200 previsões OOF |
| Teste estratificado | Patents | 0,024 | 0,033 | 0,845 | 1.000 |
| Graph transfer | Amazon → Patents | 0,131 | 0,160 | -2,616 | 5.000 |

Validação dos splits random: Amazon MAE 0,066, RMSE 0,112, R² 0,932; Patents MAE 0,026, RMSE 0,039, R² 0,803. O teste permanece fora das decisões de ajuste.

### Interpretação possível

- No split random, Amazon tem R² maior, enquanto Patents tem erros absolutos menores. Isso não significa automaticamente que Patents seja mais fácil: os alvos podem ter escalas e variâncias diferentes.
- Na CV progressiva, Amazon mantém R² médio alto (0,916); Patents tem R² médio menor (0,679), apesar de MAE/RMSE absolutos menores. Uma possível explicação é que a TDS em Patents tenha menor variância, mas a distribuição precisa ser examinada antes de concluir.
- Em ambos os grafos, o R² médio da CV progressiva é inferior ao do teste estratificado. Os protocolos respondem a perguntas diferentes: a CV avança para tamanhos maiores, enquanto o teste contém tamanhos já representados no desenvolvimento.
- A transferência Amazon → Patents tem R² negativo (-2,616), o que indica desempenho inferior a prever a média do alvo no conjunto Patents. O aumento dos erros sugere que a transferência entre estes grafos é difícil com esta representação. Mudança de distribuição é uma hipótese a investigar, não uma causa já demonstrada.
- O resultado de transferência vale para uma direção específica, Amazon → Patents; não generalizar a todos os pares de grafos.

### R² por fold nos splits progressivos

| Fold | Amazon R² | Patents R² |
|---|---:|---:|
| 1 | 0,835 | 0,579 |
| 2 | 0,940 | 0,555 |
| 3 | 0,938 | 0,711 |
| 4 | 0,934 | 0,753 |
| 5 | 0,934 | 0,796 |

O MAE cai ao longo dos folds e o R² de Patents melhora nos folds finais. Isso pode estar associado ao aumento do treino, mas os blocos de validação e a variância dos alvos também mudam; não atribuir o padrão a uma única causa.

### Interpretação das métricas

- **MAE:** erro absoluto médio, na escala da TDS.
- **RMSE:** erro quadrático médio em raiz, na escala da TDS e mais sensível a erros grandes.
- **R²:** desempenho relativo à previsão pela média; não é acurácia. Valores negativos significam desempenho inferior a essa referência.

### Visuais recomendados

1. MAE/RMSE por grafo e protocolo, com eixos e legendas explícitos.
2. R² por fold junto das faixas de tamanho de treino e validação.
3. Distribuição de TDS por grafo e por tamanho para contextualizar escalas e variâncias.
4. Real versus previsto e resíduos para teste final e transferência, caso sejam usados os CSVs de previsões.

## 10. Limitações, questões de ML e próximos passos — 3 min, mais discussão opcional

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
5. Como avaliar a contribuição de `num_vertices`, `num_edges`, `subgraph_size` e das estatísticas de grau por ablação e importância de atributos?
6. Quais métricas refletem melhor o uso futuro do modelo na otimização, além de MAE, RMSE e R²?
7. Como planejar experimentos com múltiplas sementes e grafos para medir robustez?

### Próximos passos possíveis

1. Definir explicitamente a população-alvo e o tipo de generalização desejado.
2. Incluir baselines simples e comparar modelos sob o mesmo protocolo.
3. Avaliar métricas por tamanho e inspecionar erros e resíduos.
4. Fazer análise de atributos e ablação.
5. Repetir com múltiplas sementes e grafos.
6. Avaliar a utilidade do modelo numa etapa concreta de busca do CSOP.

### Perguntas prováveis e respostas que a apresentação deve cobrir

- **Por que regressão?** O alvo TDS é numérico e contínuo. Uma etapa futura pode usar previsões para ordenar candidatos.
- **Por que TDS?** É a função objetivo atualmente implementada e permite validar o pipeline. O projeto pode receber outras funções objetivo.
- **Como são gerados os rótulos?** Calculados diretamente pela contagem de triângulos no subgrafo, normalizada pelo número de vértices.
- **O que significa cada feature?** Explicar grafo global, tamanho do candidato e estatísticas agregadas de grau; salientar que atributos globais são constantes dentro de cada grafo.
- **Há risco de vazamento?** O conjunto de teste não deve participar da seleção do modelo ou hiperparâmetros. Também discutir dependência entre candidatos que podem compartilhar vértices/arestas.
- **Por que o split random não basta?** Ele estima desempenho em amostras novas da mistura de tamanhos observada; não responde sobre tamanhos totalmente futuros nem grafos diferentes.
- **O que mede o time series?** A ordem é por tamanho, não por tempo. Descrever como validação progressiva por tamanho.
- **O teste no protocolo time_series é temporal?** Não. O experimento reserva 20% estratificados em todos os tamanhos e aplica folds progressivos nos 80% restantes. Se a pergunta exigir teste final só em tamanhos maiores, é preciso outro holdout por faixa de tamanho.
- **Por que R² difere de MAE entre grafos?** R² é relativo à variância do alvo; MAE/RMSE dependem da escala. Mostrar distribuições de TDS e baselines por grafo.
- **O modelo é rápido o suficiente?** Ainda não foi medido comparativamente. Medir extração de features e predição contra custo de calcular TDS antes de alegar benefício na busca.
- **Por que não GNN?** O escopo atual prioriza modelos tabulares compactos e potencialmente rápidos. GNNs podem ser uma comparação futura se a informação topológica resumida pelas features for insuficiente.
- **Como evitar sobreajuste ao ajustar hiperparâmetros?** Fazer qualquer seleção dentro dos dados de desenvolvimento e manter o teste intocado; considerar validação aninhada caso a busca seja extensa.

## Orientações para geração dos slides

Este planejamento será usado como entrada para uma ferramenta de geração de slides. Preservar valores e ressalvas; não inventar resultados nem tratar tamanho como tempo real. Distinguir claramente observações, interpretações possíveis e propostas futuras. Priorizar detalhes do pipeline e do processo de ML, com visuais para o fluxo, cada protocolo de split e os resultados. Evitar slides com parágrafos longos; usar notas de apoio para definições e ressalvas metodológicas.

## Referências aos artefatos dos experimentos

- Amazon, split random: [configuração](../experiments/runs/EXP_20261021_88cae057a82d/config/experiment.json), [métricas](../experiments/runs/EXP_20261021_88cae057a82d/evaluation/metrics.json), [splits](../experiments/runs/EXP_20261021_88cae057a82d/datasets/split_metadata.json).
- Patents, split random: [configuração](../experiments/runs/EXP_20261112_70f000f9ad58/config/experiment.json), [métricas](../experiments/runs/EXP_20261112_70f000f9ad58/evaluation/metrics.json), [splits](../experiments/runs/EXP_20261112_70f000f9ad58/datasets/split_metadata.json).
- Amazon, progressivo por tamanho: [configuração](../experiments/runs/EXP_20261111_307605b45817/config/experiment.json), [métricas](../experiments/runs/EXP_20261111_307605b45817/evaluation/metrics.json), [splits](../experiments/runs/EXP_20261111_307605b45817/datasets/split_metadata.json).
- Patents, progressivo por tamanho: [configuração](../experiments/runs/EXP_20261113_8990d8ddf6e9/config/experiment.json), [métricas](../experiments/runs/EXP_20261113_8990d8ddf6e9/evaluation/metrics.json), [splits](../experiments/runs/EXP_20261113_8990d8ddf6e9/datasets/split_metadata.json).
- Amazon → Patents, graph transfer: [configuração](../experiments/runs/EXP_20261104_183446919fe2/config/experiment.json), [métricas](../experiments/runs/EXP_20261104_183446919fe2/evaluation/metrics.json), [splits](../experiments/runs/EXP_20261104_183446919fe2/datasets/split_metadata.json).
