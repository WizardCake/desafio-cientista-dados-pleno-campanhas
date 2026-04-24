# Relatório de Resultados — Inteligência de Escolha para Disparos WhatsApp

> **Projeto:** Prefeitura do Rio — Squad WhatsApp
> **Data:** Abril 2026

---

## 1. Inspeção Inicial dos Dados (Notebook 00)

### Visão Geral das Bases

| Base | Registros | Colunas |
|------|-----------|---------|
| Disparos | 392.921 | 16 |
| Dimensão Telefones | 283.289 | 11 |

### Distribuição de Status

| Status | Quantidade | % |
|--------|------------|---|
| Read | 271.566 | 69,1% |
| Delivered | 85.405 | 21,7% |
| Failed | 27.257 | 6,9% |
| Sent | 5.533 | 1,4% |
| Processing | 3.160 | 0,8% |

### Divergências de Schema

- `telefone_mascarado` **não existe** na base real — a chave é `telefone_numero`
- Status `processing` **não está previsto** no schema oficial
- Colunas `resposta_datahora` e `fim_datahora` **não existem** na base

### Chave de Junção e Cobertura

- Chave: `contato_telefone` (disparos) ↔ `telefone_numero` (dimensão)
- Taxa de match: **~85,6%** (252.497 telefones na interseção)
- **42.406 telefones** em disparos sem correspondência na dimensão (**14,38%** de perda)

### Campo `telefone_aparicoes`

- Tipo: array numpy de dicionários com chaves: `id_sistema`, `cpf`, `proprietario_tipo`, `registro_data_atualizacao`
- Máximo de sistemas por telefone: **76.480** (outlier extremo)
- Mediana de sistemas por telefone: **3**

---

## 2. EDA e Qualidade de Fontes (Notebook 01)

### Ranking Causal de Sistemas (6 sistemas)

| Rank | id_sistema | Total Disparos | Taxa Entrega | EB Lower | Score |
|------|------------|----------------|-------------|----------|-------|
| 1 | -4704... | 195.363 | 99,29% | 99,25% | 1,000 |
| 2 | -2757... | 6.796 | 96,53% | 96,09% | 0,307 |
| 3 | 3094... | 242.123 | 95,61% | 95,53% | 0,184 |
| 4 | 4458... | 19.694 | 95,26% | 94,97% | 0,062 |
| 5 | -1336... | 20.434 | 95,17% | 94,89% | 0,045 |
| 6 | 1257... | 149.899 | 94,80% | 94,69% | 0,000 |

> Ranking via empirical Bayes com shrinkage proporcional ao volume. O sistema `-4704...` é claramente superior; os demais formam um bloco com diferenças marginais.

### Decaimento Temporal

| Idade do Telefone | Taxa Entrega | Taxa Leitura |
|-------------------|-------------|-------------|
| < 30 dias | 98,27% | 76,18% |
| 30–90 dias | 97,49% | 75,37% |
| 90–180 dias | 98,11% | 76,63% |
| 180 dias–1 ano | 94,68% | 71,66% |
| 1–2 anos | 94,14% | 71,12% |
| > 2 anos | 93,33% | 69,92% |

- **Chi-square:** 5.426,53; **p ≈ 0** → efeito temporal estatisticamente significativo
- Telefone recente (>180d) entrega ~4pp a mais que telefone antigo (>1ano)
- Há uma anomalia positiva na faixa 90–180d que merece investigação

### Análise de Viés de Seleção

| Métrica | Valor |
|---------|-------|
| Correlação Volume × Taxa | R = 0,233; p = 0,657 |
| Significância a 5% | **Não significativo** |

- O ranking por primeiro contato (**first-touch**) preserva a ordem dos 3 melhores sistemas, mas inverte o 4º e o 5º — indica viés moderado, porém insuficiente para alterar o champion.

### Vitórias Intra-CPF (Controlando Viés)

| Sistema | Taxa de Vitória | Participações |
|---------|----------------|---------------|
| -4704... | 91,6% | — |
| -1336... | 99,3% | 13.400 |

- O sistema `-4704...` vence em 91,6% dos confrontos intra-CPF, confirmando robustez do ranking.
- O sistema `-1336...` vence 99,3%, mas com baixa participação — resultado menos confiável.

### Estratificação por `categoria_hsm`

| Categoria | Range Taxa Entrega | Confiabilidade |
|-----------|-------------------|----------------|
| Utility | 96–99% | Alta |
| Authentication | 94–99% | Alta |
| Marketing | — | Baixa (poucos dados) |
| REMOVED | — | Baixa (poucos dados) |

- Comportamento consistente nas categorias com volume suficiente. Marketing e REMOVED não permitem conclusões.

---

## 3. Inteligência de Priorização (Notebook 02)

### Partição Temporal

| Conjunto | Registros | Proporção |
|----------|-----------|-----------|
| Train | 233.856 | 60% |
| Tuning | 77.952 | 20% |
| Test | 77.953 | 20% |

### Ranking no Treino

Mesmos 6 sistemas, mesma ordem. Range de scores: **0,000 a 1,000** — consistente com o ranking do Notebook 01.

### Grid Search — Meia-vida (Tuning)

| Meia-vida | AUC | Log Loss | Brier | Rank Médio |
|-----------|-----|----------|-------|------------|
| **90** | **0,7107** | **0,2339** | 0,0619 | **1,67** |
| 60 | 0,7100 | 0,2340 | 0,0616 | 2,33 |
| 120 | 0,7100 | 0,2342 | 0,0622 | 3,33 |
| 30 | 0,7079 | 0,2340 | 0,0612 | 3,67 |

> **Melhor meia-vida: 90 dias**

### Validação do Modelo (Test)

| Métrica | Valor |
|---------|-------|
| AUC | 0,6624 |
| Log Loss | 0,2662 |
| Brier | 0,0674 |

**Top features do modelo logístico:**

| Feature | Coeficiente |
|---------|-------------|
| melhor_score_sistema | 1,51 |
| log_n_sistemas_telefone | 0,64 |
| penalidade_proprietarios | 0,16 |

### Comparação Offline (Test)

| Método | Cobertura Holdout | Entrega Top1 | Read Top1 | Prob ≥1 Entrega |
|--------|------------------|--------------|-----------|-----------------|
| **fonte_mais_recente** | **17,85%** | **88,66%** | **63,99%** | **84,23%** |
| heuristica | 17,85% | 84,89% | 58,30% | 84,16% |
| random (média 20) | 17,88% | 83,07% | 57,47% | 84,12% |
| alfabetico | 17,89% | 79,66% | 52,73% | 84,01% |
| mais_recente | 17,91% | 77,30% | 48,40% | 83,99% |
| modelo | 17,87% | 74,26% | 44,47% | 83,99% |

> **Champion: `fonte_mais_recente`** — escolhe o telefone cuja fonte (sistema) tem o maior score e foi atualizada mais recentemente.
> Status: **candidato_para_ab_validacao_offline_fraca**

- Random **não superou** o champion; cobertura < 50%
- Modelo **não superou** baselines determinísticos

### Output Operacional

| Métrica | Valor |
|---------|-------|
| CPFs elegíveis (2+ telefones) | 30.695 |
| Total de CPFs candidatos | 1.214.718 |
| Método champion | fonte_mais_recente |

---

## 4. Desenho do Experimento (Notebook 03)

### Baseline

| Métrica | Valor |
|---------|-------|
| CPFs com histórico de disparo | 267.116 |
| CPFs elegíveis na dimensão (2+ telefones) | 30.695 |
| Elegíveis com disparo | 10.411 (3,9% de cobertura) |
| Taxa de entrega (total) | 96,47% |
| Taxa de entrega (últimos 28d) | 95,93% |

### Tamanho Amostral (α = 0,05; poder = 0,80)

| Uplift detectável | CPFs necessários | Dias estimados |
|--------------------|------------------|-----------------|
| 0,5 pp | 36.318 | 2.588 |
| 1,0 pp | 8.476 | 604 |
| 2,0 pp | 1.792 | 128 |
| 3,0 pp | 625 | 45 |

> Com ~14 CPFs elegíveis/dia, testes A/B tornam-se intratáveis para efeitos pequenos.

---

## Pontos de Atenção Críticos

### 1. Modelo não superou baselines simples

O AUC do modelo logístico (0,66) ficou **abaixo** de métodos determinísticos como `fonte_mais_recente`. O champion é uma regra simples, não um modelo preditivo. A complexidade adicional do modelo não justifica o ganho — ou melhor, a **ausência** de ganho.

### 2. Cobertura do holdout muito baixa (~18%)

A maioria dos CPFs no conjunto de teste **não possui dados históricos por telefone**, tornando as métricas offline **proxies frágeis** da performance real. Os resultados devem ser interpretados com cautela.

### 3. Diferenças marginais entre métodos

As diferenças entre todos os métodos são **sub-percentuais** nas métricas proxy. Isso sugere que o ganho real da otimização pode ser **mínimo** e possivelmente dentro do ruído estatístico.

### 4. Baseline muito alto (96% entrega)

Com taxa de entrega tão alta para CPFs elegíveis, o espaço para melhoria é **estreito**. Métricas como **leitura** ou **custo por mensagem lida** podem ser mais discriminativas e alinhadas ao valor de negócio.

### 5. Volume elegível reduzido

Apenas ~14 CPFs elegíveis/dia tornam testes A/B **intratáveis** para detecção de efeitos pequenos. Isso limita substancialmente a validação experimental.

### 6. Teste A/B de longuíssima duração

Para detectar um uplift de **1 pp** seriam necessários ~604 dias — inviável operacionalmente. Efeitos de 2 pp ou mais são mais realistas para experimentação.

---

## Conclusão

O sistema `fonte_mais_recente` — uma heurística simples que seleciona o telefone cuja fonte tem o melhor score histórico e foi atualizada mais recentemente — é o **champion** identificado. No entanto, os ganhos em relação a baselines aleatórios são marginais, e o modelo preditivo não agrega valor sobre regras determinísticas.

O principal gargalo para validação e impacto é a **baixa cobertura de CPFs elegíveis** (3,9% do universo), que torna tanto a medição offline quanto a experimentação A/B extremamente limitadas. Recomenda-se focar em **expandir o universo elegível** (e.g., relaxar critérios de qualificação) e/ou **redefinir a métrica de sucesso** para leitura ou custo-effetiveness em vez de entrega.