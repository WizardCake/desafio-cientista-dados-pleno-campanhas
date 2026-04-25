# Relatório Consolidado da Solução

## 1. Visão geral

Este relatório consolida, em uma narrativa única, o raciocínio desenvolvido nos quatro notebooks da solução. O objetivo do desafio é criar uma **inteligência de escolha de telefones para disparos via WhatsApp**: a partir de um CPF com múltiplos telefones candidatos, selecionar automaticamente os dois telefones com maior chance de gerar contato efetivo com o cidadão.

A solução foi construída em quatro etapas encadeadas:

1. **Inspecionar e validar os dados**, para garantir que as chaves, schemas, status e campos relevantes permitiam uma análise confiável.
2. **Medir a qualidade das fontes de telefone**, atribuindo desempenho aos sistemas de origem de forma causal, sem dar crédito a uma fonte que só apareceu depois do disparo.
3. **Transformar os sinais analíticos em uma regra operacional**, capaz de ranquear telefones por CPF e escolher os dois melhores candidatos.
4. **Desenhar um experimento A/B**, porque a evidência offline é útil para propor uma regra, mas não é suficiente para provar ganho causal em produção.

O ponto central da solução é que não basta calcular uma taxa média de entrega por fonte. É necessário responder a uma sequência lógica: qual telefone veio de qual sistema, se essa origem já existia antes do envio, se a idade do dado reduz a chance de sucesso, se o modelo melhora a decisão operacional e como validar essa decisão em produção.

---

## 2. Problema de negócio

A Prefeitura envia milhares de mensagens via WhatsApp. Cada disparo tem custo, e a janela de comunicação com o cidadão é limitada. O RMI consolida telefones oriundos de diferentes sistemas municipais, mas um mesmo CPF pode ter vários telefones associados, com qualidades e atualizações diferentes.

O desafio, portanto, não é apenas encontrar telefones válidos. É definir uma política de priorização:

- quais fontes de dados são mais confiáveis;
- quanto a atualidade do telefone importa;
- como combinar origem, recência, DDD e outros sinais em uma regra prática;
- como escolher dois telefones por CPF;
- como validar que essa regra é melhor que a regra atual.

A solução proposta trata o problema como uma decisão operacional orientada por evidência: primeiro estima a confiabilidade das fontes, depois usa esses sinais para ranquear telefones e, por fim, propõe um teste A/B para validação causal.

---

## 3. Definição da métrica principal

A métrica operacional principal usada nos notebooks é **entrega**, definida como:

```text
entrega = status_disparo em {delivered, read}
```

O enunciado menciona taxa de entrega (`DELIVERED`). No log real, também existe o status `read`. A interpretação adotada é que `read` representa um estado posterior e mais forte que `delivered`: se a mensagem foi lida, ela necessariamente chegou ao WhatsApp do destinatário. Por isso, `read` é contabilizado como evidência de entrega.

A taxa de leitura (`read`) é mantida separadamente como métrica secundária, pois está mais próxima do valor de negócio do contato efetivo. Assim, a análise separa dois conceitos:

- **entrega operacional:** `delivered` ou `read`;
- **leitura:** apenas `read`.

Essa definição é usada de forma consistente nos quatro notebooks.

---

## 4. Notebook 00: inspeção inicial e premissas de dados

O primeiro notebook não tenta resolver a priorização. Sua função é garantir que a base esteja suficientemente compreendida para sustentar as análises posteriores.

### 4.1 Divergências entre schema e dados reais

A inspeção mostrou que havia diferenças relevantes entre o schema documentado e os arquivos Parquet reais:

| Tema | Observação | Decisão |
|---|---|---|
| Chave da dimensão | O schema sugeria `telefone_mascarado`, mas a coluna real é `telefone_numero` | Usar `telefone_numero` no join |
| Chave do disparo | A base de disparos usa `contato_telefone` | Fazer join `contato_telefone == telefone_numero` |
| Status | Além de `sent`, `delivered`, `read`, `failed`, existe `processing` | Excluir `processing` das análises de desfecho |
| Colunas ausentes | `resposta_datahora` e `fim_datahora` não existem no Parquet | Ignorar essas colunas |
| Campo de origem | `telefone_aparicoes` usa `id_sistema`, não `id_sistema_mask` | Usar `id_sistema` como identificador da fonte |

Essa etapa é importante porque a atribuição de desempenho às fontes depende diretamente de um join correto. Se a chave estiver errada, todo ranking de sistemas fica comprometido.

### 4.2 Cobertura do join

A chave operacional validada foi:

```text
base_disparo.contato_telefone == dim_telefone.telefone_numero
```

Como os telefones estão mascarados/hasheados, a análise de magnitude não tenta interpretar números reais. Ela verifica apenas se as duas colunas estão na mesma escala de codificação e mede a interseção entre elas.

O resultado foi relevante: cerca de **85,6%** dos telefones da base de disparos possuem correspondência na dimensão. Isso significa que uma parte dos disparos não pode receber atribuição de fonte, limitando a cobertura do ranking causal.

### 4.3 Campo crítico: `telefone_aparicoes`

O campo mais importante da dimensão é `telefone_aparicoes`. Ele é um array de dicionários e contém:

- sistema de origem do telefone (`id_sistema`);
- CPF associado naquela origem;
- tipo de proprietário;
- data de atualização do registro.

Esse campo permite conectar o telefone a suas fontes históricas. Sem desestruturar esse array, não seria possível medir o desempenho de cada sistema de origem.

### 4.4 Premissas estabelecidas

A inspeção inicial fixou as premissas que orientam o restante da solução:

- `processing` é status intermediário e não entra no cálculo de desfecho;
- telefones fixos são excluídos por premissa operacional de WhatsApp;
- entrega operacional é `delivered` ou `read`;
- a baixa sobreposição de CPFs entre dimensão e logs limita validações diretas no nível do cidadão;
- o join entre disparos e fontes é possível, mas não cobre 100% dos eventos.

Com essas premissas, os notebooks seguintes partem de uma base compreendida e tratam a pergunta substantiva: quais fontes parecem mais confiáveis?

---

## 5. Notebook 01: qualidade das fontes e decaimento temporal

O notebook 01 transforma a inspeção inicial em análise de qualidade das fontes. O objetivo é medir o "calor" de cada sistema, isto é, a probabilidade de entrega associada aos telefones que vieram daquela origem.

### 5.1 Por que o join precisa ser causal

Um telefone pode aparecer em vários sistemas e em diferentes datas. Se simplesmente associássemos cada disparo a todos os sistemas presentes na dimensão, haveria risco de atribuir crédito a uma fonte que só registrou aquele telefone depois do envio.

Por isso, a solução usa um **join causal**:

```text
uma fonte só recebe crédito por um disparo se
registro_data_atualizacao <= envio_datahora
```

Essa decisão evita vazamento temporal. O sistema só pode ser recompensado se aquela informação já existia no momento em que o disparo ocorreu.

### 5.2 Deduplicação de aparições

O array `telefone_aparicoes` é explodido para gerar uma linha por aparição bruta. Em seguida, as aparições são deduplicadas por:

```text
(telefone_numero, id_sistema, registro_data_atualizacao)
```

A deduplicação evita que uma fonte pareça melhor apenas por ter várias linhas repetidas para o mesmo telefone no mesmo sistema. A lógica é preservar a linha do tempo sem inflar artificialmente o peso de uma fonte.

### 5.3 Ranking de fontes

Para cada sistema de origem, a métrica bruta é:

```text
taxa_entrega = (read + delivered) / total_disparos
```

Entretanto, a taxa bruta pode favorecer fontes pequenas. Uma fonte com poucos disparos pode ter taxa alta por acaso. Para reduzir esse problema, o ranking principal usa **Empirical Bayes Lower Bound**, um limite inferior beta-binomial empírico.

A lógica é:

- fontes com muito volume ficam mais próximas da sua taxa observada;
- fontes com pouco volume são encolhidas em direção à média global;
- o ranking fica mais conservador e menos sensível a flutuações de amostra pequena.

O Wilson lower bound também é calculado como diagnóstico complementar, mas o ranking principal usa Empirical Bayes.

### 5.4 Resultado do ranking

O ranking causal principal encontrou seis sistemas com volume relevante. As taxas de entrega são altas em todos eles, variando aproximadamente entre **94% e 99%**. O sistema `-4704067261970591609` aparece como a fonte mais forte, com taxa de entrega próxima de 99% e maior limite inferior Empirical Bayes.

Essa diferença é estatisticamente útil para ranquear, mas a leitura de negócio precisa ser cautelosa: como todos os sistemas têm entrega alta, o ganho prático de trocar a regra pode ser pequeno.

### 5.5 Testes de viés de seleção

O enunciado alerta que algumas bases podem aparecer mais nos logs porque já eram consideradas "quentes". Isso cria risco de viés de seleção: uma fonte pode parecer melhor não por qualidade intrínseca, mas por ter recebido melhores oportunidades historicamente.

Para avaliar isso, foram feitos três testes:

1. **Correlação entre volume e taxa:** verifica se fontes com mais disparos têm taxa artificialmente superior.
2. **Full attribution vs. first-touch:** compara o ranking quando todas as fontes causais recebem crédito contra o cenário em que apenas a fonte mais recente recebe crédito.
3. **Vitórias intra-CPF:** compara fontes competindo dentro do mesmo CPF, reduzindo vieses de composição entre cidadãos.

Esses testes não eliminam todo viés, mas ajudam a verificar se o ranking é apenas um artefato de volume ou atribuição. A conclusão foi que o ranking é relativamente robusto, embora existam sinais de assimetria intra-CPF e diferenças práticas pequenas.

### 5.6 Decaimento temporal

A análise também avaliou se a idade do dado afeta a chance de entrega. A pergunta era: um telefone atualizado recentemente é mais "quente" que um telefone antigo?

O resultado mostrou queda de qualidade para dados mais antigos. A taxa de entrega fica em torno de **98%** para registros com menos de 180 dias e cai para cerca de **93%** quando o dado tem mais de 2 anos. A relação não é perfeitamente monotônica, mas o padrão geral indica perda de valor com o tempo.

Essa evidência justifica incluir recência no algoritmo. No notebook 02, essa recência entra por decaimento exponencial, com half-life escolhido no tuning.

---

## 6. Notebook 02: inteligência de priorização

O notebook 02 transforma a análise em regra operacional. A pergunta muda: dado um CPF com vários telefones, quais dois devem ser escolhidos para envio?

### 6.1 Separação temporal

O ranking de sistemas é aprendido apenas com dados de treino. O restante do histórico é separado em tuning e teste temporal. Essa escolha reduz vazamento, pois o algoritmo não pode usar informação futura para avaliar decisões passadas.

A lógica temporal aparece em três pontos:

- o ranking de sistemas usa apenas disparos do período de treino;
- o half-life da recência é escolhido no período de tuning;
- a comparação final entre métodos usa holdout temporal.

Essa separação não transforma a validação offline em prova causal, mas melhora a disciplina metodológica.

### 6.2 Score de origem e recência

A origem do dado entra como `score_sistema`, derivado do ranking Empirical Bayes. A atualidade entra por decaimento exponencial:

```text
decaimento = exp(-ln(2) * dias_desde_atualizacao / half_life)
```

O score da aparição combina origem e tempo:

```text
score_aparicao = score_sistema * decaimento
```

Assim, uma fonte muito boa perde força se o registro for antigo. Da mesma forma, uma fonte mais fraca não ganha prioridade apenas por ser recente, pois a origem também pesa.

### 6.3 Features usadas na decisão

O algoritmo combina sinais de diferentes naturezas:

| Dimensão | Sinal |
|---|---|
| Origem | score Empirical Bayes do sistema |
| Atualidade | dias desde atualização e decaimento temporal |
| Geografia | DDD e prior suavizado por DDD |
| Qualidade cadastral | `telefone_qualidade` |
| Risco de compartilhamento | quantidade de proprietários e CPFs distintos por telefone |
| Robustez de origem | quantidade de sistemas candidatos e proporção de aparições causais |

Essas features alimentam dois caminhos:

- um modelo logístico, usado como modelo preditivo de entrega;
- uma heurística ponderada, usada como baseline interpretável.

Além disso, foram comparadas regras determinísticas simples, como telefone mais recente, fonte mais recente, ordem alfabética e seleção aleatória.

### 6.4 Separar predição de decisão

Um ponto importante do notebook 02 é a separação entre duas perguntas:

1. O modelo consegue prever entrega?
2. Usar esse score para escolher os dois telefones por CPF melhora a política de envio?

O modelo logístico apresentou AUC em torno de **0,66**, indicando alguma capacidade de ordenação. Porém, quando usado como política de escolha top-2, não superou os baselines determinísticos simples.

Isso é uma conclusão relevante: um modelo pode prever razoavelmente bem no nível do evento e ainda assim não ser a melhor política operacional para selecionar telefones.

### 6.5 Champion escolhido

Na validação offline, a política que melhor se comportou foi `fonte_mais_recente`. Ela prioriza o telefone cuja fonte causal mais recente possui maior score, usando recência e score heurístico como desempates.

O champion escolhido, portanto, é uma regra determinística:

```text
champion = fonte_mais_recente
```

Essa escolha é pragmática. Em vez de defender um modelo mais complexo sem ganho claro, a solução seleciona uma regra simples, explicável e aderente aos sinais aprendidos na EDA.

### 6.6 Por que o champion ainda não é uma prova final

A validação offline tem limitações importantes:

- a cobertura do holdout por telefone é baixa, em torno de 18%;
- as diferenças entre métodos são pequenas, em muitos casos menores que 1 ponto percentual;
- o histórico observado reflete a política antiga de disparo;
- há viés de seleção e baixa capacidade de observar contrafactuais.

Por isso, o notebook 02 não afirma que o champion venceu definitivamente. Ele afirma que `fonte_mais_recente` é a melhor política candidata para ser testada em produção.

---

## 7. Notebook 03: desenho do experimento

O notebook 03 fecha a solução mudando o padrão de evidência. Os notebooks anteriores constroem uma política candidata com base em histórico, mas histórico não prova que a nova política causará ganho em produção.

A validação final deve ser experimental.

### 7.1 Unidade de randomização

A unidade de randomização proposta é o **CPF**, não o telefone.

Isso é essencial porque a decisão operacional acontece no nível do CPF: escolher dois telefones candidatos para aquele cidadão. Se telefones do mesmo CPF fossem distribuídos entre controle e tratamento, haveria contaminação entre políticas.

### 7.2 População elegível

A população principal do teste são CPFs com dois ou mais telefones móveis candidatos. CPFs com apenas um telefone continuam relevantes para cobertura geral, mas não entram na análise principal porque não há decisão de priorização a validar.

Na base analisada, a população elegível é restrita:

- cerca de **30.695 CPFs** na dimensão RMI têm dois ou mais telefones;
- cerca de **10.411 CPFs** elegíveis tiveram disparos no período histórico;
- o volume diário estimado é de aproximadamente **14 CPFs elegíveis por dia**.

Esse baixo volume é uma restrição central do experimento.

### 7.3 Hipóteses

As hipóteses do teste são:

```text
H0: a regra champion não aumenta a proporção de CPFs elegíveis com pelo menos uma entrega.
H1: a regra champion aumenta a proporção de CPFs elegíveis com pelo menos uma entrega.
```

A métrica primária é:

```text
CPF elegível teve pelo menos uma entrega entre os telefones selecionados
```

Novamente, entrega significa `delivered` ou `read`.

### 7.4 Métricas secundárias e guardrails

As métricas secundárias propostas incluem:

- CPF teve pelo menos um `read`;
- taxa de falha por CPF;
- custo por entrega;
- cobertura elegível;
- taxa por telefone enviado.

Os guardrails monitoram riscos operacionais e de qualidade:

- bloqueios/spam;
- incidentes de API ou fornecedor;
- desequilíbrio por categoria de campanha;
- DDD;
- quantidade de telefones candidatos;
- múltiplos proprietários;
- CPFs distintos por telefone.

Esses guardrails são importantes porque uma política pode aumentar entrega média e, ainda assim, piorar segmentos sensíveis ou aumentar risco de contato com destinatário errado.

### 7.5 Tamanho amostral e duração

O baseline de entrega já é alto, em torno de **96%** para CPFs elegíveis. Isso reduz o espaço para ganho marginal. Além disso, o volume diário elegível é baixo. Como consequência, detectar efeitos pequenos exigiria muito tempo.

Estimativas do notebook:

| Uplift mínimo detectável | CPFs necessários | Duração estimada |
|---|---:|---:|
| 0,5 p.p. | ~36.318 | ~2.588 dias |
| 1,0 p.p. | ~8.476 | ~604 dias |
| 2,0 p.p. | ~1.792 | ~128 dias |
| 3,0 p.p. | ~625 | ~45 dias |

A conclusão é que um A/B para detectar ganhos pequenos em entrega é impraticável com o volume observado. O experimento é mais viável se:

- o efeito esperado for maior;
- a população elegível for ampliada;
- a métrica primária for mais discriminativa, como leitura ou custo por mensagem lida.

---

## 8. Produto final proposto

A solução entrega uma política candidata para priorização de telefones:

1. Explodir `telefone_aparicoes` e associar cada telefone aos sistemas de origem.
2. Usar apenas aparições causalmente disponíveis antes do disparo ou do momento de decisão.
3. Calcular score de confiabilidade por sistema com Empirical Bayes.
4. Aplicar decaimento temporal ao score da origem.
5. Combinar sinais de origem, recência, DDD, qualidade e risco de compartilhamento.
6. Para cada CPF elegível, escolher dois telefones.
7. Usar como champion a política `fonte_mais_recente`, por ser simples e ter melhor comportamento offline.
8. Validar em produção por teste A/B randomizado por CPF.

O resultado operacional esperado é uma tabela `resultado_escolha` com:

- CPF;
- `telefone_1`;
- `telefone_2`;
- scores auxiliares;
- método champion;
- status da evidência offline.

---

## 9. Principais limitações

A solução é deliberadamente cautelosa. Os principais limites são:

1. **Cobertura incompleta do join:** nem todos os telefones disparados possuem correspondência na dimensão.
2. **Baixa sobreposição de CPFs:** a validação direta por CPF entre dimensão e logs é limitada.
3. **Viés de seleção:** o histórico reflete decisões anteriores de envio, não uma exposição aleatória das fontes.
4. **Holdout frágil:** poucos telefones escolhidos têm evidência histórica individual suficiente.
5. **Diferenças pequenas entre métodos:** o ganho offline é marginal.
6. **Baseline de entrega alto:** há pouco espaço para melhorar a métrica de entrega.
7. **Volume elegível baixo:** o A/B pode ser longo para efeitos pequenos.
8. **Heurística não calibrada:** os pesos heurísticos são baseline comparativo, não estimativa causal de importância.

Essas limitações não invalidam a solução. Elas definem seu grau correto de confiança: a política é uma candidata bem fundamentada, não uma regra que deve substituir a atual sem experimento.

---

## 10. Conclusão

A solução constrói um fluxo completo entre análise de dados e decisão operacional.

Primeiro, valida a estrutura real das bases e explicita as premissas necessárias para análise. Depois, mede a confiabilidade das fontes com um join causal, evitando atribuir mérito a sistemas que não tinham o telefone antes do envio. Em seguida, mostra que a atualidade do dado importa e incorpora esse sinal em um score operacional. Na etapa de priorização, compara modelo, heurística e regras simples, chegando a uma política champion determinística: `fonte_mais_recente`.

A conclusão mais importante não é apenas qual sistema aparece em primeiro lugar. O principal resultado é o nexo lógico do produto:

```text
fonte confiável + dado causalmente disponível + registro recente + menor risco cadastral
=> telefone mais promissor
=> top-2 por CPF
=> validação por A/B antes de virar política definitiva
```

Assim, a solução atende às três partes do desafio: mede o calor das fontes, transforma esse aprendizado em uma regra de escolha acionável e propõe um desenho experimental para validar o ganho em produção.
