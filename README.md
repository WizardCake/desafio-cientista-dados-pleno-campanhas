# Desafio Técnico — Cientista de Dados Pleno (Squad WhatsApp)

> **Objetivo:** Criar a **Inteligência de Escolha** — identificar quais fontes de dados são mais confiáveis e, para cada CPF com múltiplos telefones, selecionar automaticamente os **2 melhores** para receber mensagens via WhatsApp.

---

## Estrutura do Repositório

```
desafio-cientista-dados-pleno-campanhas/
├── README.md                               # Este arquivo
├── RELATORIO.md                            # Relatório completo de resultados
├── requirements.txt                        # Dependências Python
├── tests/
│   └── test_utils.py                       # Testes unitários
├── src/
│   └── utils.py                            # Funções utilitárias compartilhadas
├── notebooks/
│   ├── 00_inspecao_inicial.ipynb           # Validação de schema e premissas
│   ├── 01_eda_e_qualidade_fontes.ipynb     # Parte 1: Análise exploratória e qualidade
│   ├── 02_inteligencia_priorizacao.ipynb   # Parte 2: Ranking e algoritmo de escolha
│   └── 03_desenho_experimento.ipynb        # Parte 3: Desenho de teste A/B
├── enunciado/
│   └── README.md                           # Enunciado original do desafio
└── data/
    ├── raw/                                # Dados brutos (baixados do GCS, .gitignore)
    └── processed/                          # Artefatos gerados (.gitignore)
```

---

## Como Reproduzir

### 1. Instalar Dependências

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Baixar os Dados

Os dados são baixados automaticamente no Notebook 00 a partir do bucket GCS público:
`https://storage.googleapis.com/case_vagas/whatsapp/`

Ou manualmente seguindo as instruções no notebook.

### 3. Executar os Notebooks (em ordem)

1. `00_inspecao_inicial.ipynb` — Validação de schema, chaves de join e distribuições
2. `01_eda_e_qualidade_fontes.ipynb` — Parte 1 do enunciado (EDA e qualidade de fontes)
3. `02_inteligencia_priorizacao.ipynb` — Parte 2 (ranking de sistemas e algoritmo de escolha)
4. `03_desenho_experimento.ipynb` — Parte 3 (desenho de teste A/B)

### 4. Rodar os Testes

```bash
python -m pytest tests/test_utils.py -v
```

---

## Abordagem e Premissas

### Parte 1: Análise Exploratória e Qualidade de Fontes

**Desestruturação e correlação de sistemas com performance real**

O campo `telefone_aparicoes` (array de dicionários) é explodido para associar cada telefone aos seus sistemas de origem (`id_sistema`). O join é **causal**: uma fonte só recebe crédito por um disparo se o telefone já existia naquela fonte **antes** do envio. Isso evita inflar o ranking de fontes que simplesmente apareceram depois do disparo.

A deduplicação por `(telefone_numero, id_sistema)` remove múltiplas aparições do mesmo telefone no mesmo sistema, garantindo que cada fonte conte apenas uma vez por telefone.

**Métrica primária:** Taxa de entrega (`delivered` + `read`) como métrica operacional. Taxa de leitura (`read`) como métrica secundária de negócio.

**Ranking de fontes — Empirical Bayes Lower Bound**

Em vez de usar taxa bruta (que favorece fontes com pouco volume), adotamos o limite inferior beta-binomial empírico (Empirical Bayes), que encolhe fontes com pouco volume para a média global. Wilson Score é calculado como diagnóstico complementar.

**Ranking causal por atribuição full (resultado principal)**

| Rank | id_sistema           | Disparos | Taxa Entrega | EB Lower | Score |
|------|----------------------|----------|-------------|----------|-------|
| 1    | -4704067261970591609 | 195.363  | 99.29%      | 99.25%   | 1.000 |
| 2    | -2757366171786647144 | 6.796    | 96.53%      | 96.09%   | 0.307 |
| 3    | 3094574413675758272  | 242.123  | 95.61%      | 95.53%   | 0.184 |
| 4    | 4458959843028638627  | 19.694   | 95.26%      | 94.97%   | 0.062 |
| 5    | -133612832286195827  | 20.434   | 95.17%      | 94.89%   | 0.045 |
| 6    | 1257277410380486863  | 149.899  | 94.80%      | 94.69%   | 0.000 |

O ranking se sustenta nas três análises de viés (correlação volume-taxa, first-touch e intra-CPF), embora a diferenciação prática entre os sistemas seja pequena (~94-99%).

**Janela de atualidade — Decaimento temporal**

| Faixa     | Taxa Entrega | Taxa Leitura | n        |
|-----------|-------------|-------------|----------|
| <30d      | 98.27%      | 76.18%      | 7.230    |
| 30-90d    | 97.49%      | 75.37%      | 28.430   |
| 90-180d   | 98.11%      | 76.63%      | 216.459  |
| 180d-1a   | 94.68%      | 71.66%      | 96.348   |
| 1-2a      | 94.14%      | 71.12%      | 86.415   |
| >2a       | 93.33%      | 69.92%      | 74.397   |

Existe decaimento a partir de ~180 dias, com queda de ~5pp em entrega de <180d para >2a. O half-life de 90 dias (escolhido via grid search temporal) equivale a um "prazo de validade" prático de ~3 meses.

### Parte 2: Inteligência de Priorização

**Ranking de sistemas** — O mesmo ranking da Parte 1, aprendido no período de treino (60%), serve de base para o score operacional.

**Algoritmo de escolha** — Combina:
1. **Score da origem** (Empirical Bayes do sistema) × **decaimento temporal** (exponencial com half-life=90d)
2. **Score do DDD** (prior bayesiano suavizado por DDD)
3. **Qualidade do telefone** (ALTA=1.0, MEDIA=0.5, BAIXA=0.25)
4. **Exclusividade CPF-telefone** (1 / CPFs distintos por telefone)
5. **Penalidade de proprietários** (1 / n proprietários)
6. **Quantidade de sistemas** (log1p)
7. **Proporção de aparições causais**

O modelo logístico (AUC 0.66, test) e uma heurística com pesos arbitrários são comparados contra baselines determinísticos (telefone mais recente, fonte mais recente, alfabético) e um baseline aleatório.

**Resultado: o modelo NÃO superou baselines determinísticos.**

| Método                | Entrega Top1 | Read Top1 | Prob ≥1 Entrega |
|-----------------------|-------------|-----------|-----------------|
| fonte_mais_recente    | 88.66%      | 63.99%    | 84.23%          |
| heuristica            | 84.89%      | 58.30%    | 84.16%          |
| random (média 20)     | 83.07%      | 57.47%    | 84.12%          |
| alfabetico            | 79.66%      | 52.73%    | 84.01%          |
| mais_recente          | 77.30%      | 48.40%    | 83.99%          |
| modelo_logistico      | 74.26%      | 44.47%    | 83.99%          |

O champion escolhido foi **fonte_mais_recente** (telefone cuja fonte mais recente tem o maior score), com status de **candidato para validação A/B com evidência offline fraca**, pois:
- A cobertura do holdout é <20%
- As diferenças entre métodos são sub-percentuais
- O modelo não superou métodos simples

**Algoritmo operacional final:** Para cada CPF com 2+ telefones candidatos, selecionar os 2 telefones com maior `score_fonte_mais_recente`, desempatando por recência e score heurístico.

### Parte 3: Desenho de Experimento (Teste A/B)

**Hipóteses**
- H0: A regra champion não aumenta a proporção de CPFs elegíveis com pelo menos uma entrega
- H1: A regra champion aumenta essa proporção

**População:** CPFs com 2+ telefones móveis candidatos (~30.695 na dimensão, ~10.411 com histórico de disparos)

**Tamanho amostral (α=0.05, poder=0.80)**

| Uplift mínimo | CPFs necessários | Duração estimada |
|---------------|-------------------|------------------|
| 0.5 p.p.      | 36.318            | ~2.588 dias (7+ anos) |
| 1.0 p.p.      | 8.476             | ~604 dias (1.7 anos) |
| 2.0 p.p.      | 1.792             | ~128 dias (4 meses) |
| 3.0 p.p.      | 625               | ~45 dias (6 semanas) |

**Recomendação:** Considerar métricas mais discriminativas (taxa de leitura, custo por mensagem lida) e ampliar a população elegível via integração de mais fontes.

---

## Premissas

1. **Status `processing` excluído** — status intermediário sem resultado final definitivo
2. **Telefones fixos excluídos** — WhatsApp não entrega em linhas fixas (apenas 8 registros)
3. **Entrega como métrica primária operacional** — `delivered` ou `read` indicam contato efetivo
4. **Join causal** — uma fonte só recebe crédito se o telefone já existia naquela fonte antes do envio
5. **Empirical Bayes para ranking** — shrinkage bayesiano penaliza fontes com pouco volume
6. **Split temporal** — treino/tuning/teste por timestamp, não aleatório, para evitar vazamento
7. **Randomização por CPF** — evita contaminação entre grupos no A/B test

---

## Limitações e Pontos de Atenção

1. **Modelo logístico não superou baselines simples** — AUC de 0.66 ficou abaixo do método "fonte_mais_recente". O champion é determinístico, não preditivo.
2. **Cobertura do holdout baixa (~18%)** — A maioria dos CPFs no teste não tem dados per-telefone, tornando métricas offline proxies frágeis.
3. **Diferenças marginais entre métodos** — Sub-percentuais em proxy metrics reforçam necessidade de validação A/B.
4. **Baseline muito alto (96% entrega)** — Pouco espaço para ganho marginal em entrega; leitura ou custo podem ser mais discriminativos.
5. **Volume elegível reduzido** — Apenas ~14 CPFs elegíveis/dia tornam A/B tests longos para efeitos pequenos.
6. **Heurística com pesos arbitrários** — Não são calibrados estatisticamente; servem como baseline comparativo.
7. **Campo `validacao_telefone`** disponível mas não utilizado como feature — potencial para investigação futura.
8. **CPF mismatch entre tabelas** — Baixa interseção (1.331 CPFs) entre dimensão e dispatch limita validação por CPF.

---

## Tecnologias

- Python 3.12, pandas, numpy, pyarrow
- matplotlib, seaborn (visualização)
- scipy, statsmodels (estatística)
- scikit-learn (Regressão Logística)
- Jupyter Notebooks, pytest

---

## Autor

Matheus de Andrade Santos