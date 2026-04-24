# Desafio Cientista de Dados Pleno - Squad WhatsApp

Solução para o desafio técnico de Cientista de Dados Pleno - Prefeitura do Rio.

> Objetivo: criar a "Inteligência de Escolha" que seleciona os 2 melhores telefones
> para cada CPF, maximizando a chance de entrega e leitura de mensagens via WhatsApp.

---

## Estrutura do Repositório

```
desafio-cientista-dados-pleno-campanhas/
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências Python
├── tests/                              # Testes unitários
│   └── test_utils.py                   # Testes das funções de utils.py
├── src/
│   └── utils.py                        # Funções utilitárias compartilhadas
├── notebooks/
│   ├── 00_inspecao_inicial.ipynb       # Validação de premissas sobre os dados
│   ├── 01_eda_e_qualidade_fontes.ipynb # Parte 1: Análise Exploratória
│   ├── 02_inteligencia_priorizacao.ipynb # Parte 2: Algoritmo de Escolha
│   └── 03_desenho_experimento.ipynb    # Parte 3: Teste A/B
└── data/
    ├── raw/                            # Dados brutos (baixados do GCS, .gitignore)
    └── processed/                      # Artefatos intermediários (.gitignore)
```

---

## Como Reproduzir

### 1. Instalar Dependências

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Baixar os Dados

Os dados são baixados automaticamente no Notebook 00 a partir do bucket GCS público:
`https://storage.googleapis.com/case_vagas/whatsapp/`

Ou manualmente:
```bash
# Base de disparos
curl -O https://storage.googleapis.com/case_vagas/whatsapp/base_disparo_mascarado
mv base_disparo_mascarado data/raw/whatsapp_base_disparo_mascarado.parquet

# Dimensão de telefones
curl -O https://storage.googleapis.com/case_vagas/whatsapp/dim_telefone_mascarado
mv dim_telefone_mascarado data/raw/whatsapp_dim_telefone_mascarado.parquet

# Schema
curl -O https://storage.googleapis.com/case_vagas/whatsapp/schema.yml
mv schema.yml data/raw/whatsapp_schema.yml
```

### 3. Executar os Notebooks

Executar em ordem:
1. `notebooks/00_inspecao_inicial.ipynb` — valida premissas e schema
2. `notebooks/01_eda_e_qualidade_fontes.ipynb` — Parte 1 do enunciado
3. `notebooks/02_inteligencia_priorizacao.ipynb` — Parte 2 do enunciado
4. `notebooks/03_desenho_experimento.ipynb` — Parte 3 do enunciado

### 4. Rodar os Testes

```bash
python -m pytest tests/test_utils.py -v
```

---

## Resumo da Solução

### Parte 1: EDA e Qualidade de Fontes

- **Explosão do array** `telefone_aparicoes` para associar cada disparo aos sistemas de origem
- **Métrica operacional primária:** Taxa de Entrega (`delivered` ou `read`); leitura fica como métrica secundária de negócio
- **Ranking de sistemas** usando **Wilson Lower Bound** (limite inferior do IC 95%), que penaliza sistemas com pouco volume
- **Decaimento temporal:** análise de como a idade do dado impacta a chance de sucesso (com teste chi-square)
- **Análise estratificada por categoria_hsm:** verificação de que o ranking se sustenta entre categorias de campanha
- **3 análises de viés de seleção:**
  1. Correlação volume vs. taxa
  2. Análise first-touch
  3. Comparação intra-CPF

### Parte 2: Inteligência de Priorização

- **Score/modelo operacional** combina qualidade da origem, recência do dado, qualidade do telefone, DDD, quantidade de proprietários e quantidade de sistemas associados ao telefone
- **`telefone_qualidade`** (ALTA/MÉDIA/BAIXA) incluída como feature ordinal (`score_qualidade`) tanto no modelo logístico quanto na heurística
- **Half-life temporal** escolhido por validação temporal em grid, evitando fixar arbitrariamente o prazo de validade do telefone
- **Heurística com pesos arbitrários:** os pesos da heurística (0.45 origem+tempo, 0.20 DDD, 0.15 proprietários, 0.10 qualidade, 0.05 sistemas, 0.05 causalidade) foram escolhidos com base em entendimento de importância como baseline comparativo, não derivados de calibração estatística. Servem como referência contra a qual o modelo logístico é comparado.
- **Champion** escolhido no holdout comparando modelo logístico, heurística, telefone mais recente, regra alfabética e baseline aleatório
- **Algoritmo de escolha:** para cada CPF elegível, seleciona os 2 telefones de acordo com o método champion
- **Limitação do holdout:** o grid de half-life e a comparação entre métodos compartilham o mesmo set de validação, o que pode inflacionar estimativas offline. A decisão final fica para o A/B test.

### Parte 3: Desenho do Experimento

- **Hipóteses:** H0 (sem efeito) vs. H1 (aumento na taxa de entrega por CPF elegível)
- **Unidade de randomização:** CPF (evita contaminação entre grupos)
- **População principal:** CPFs com 2 ou mais telefones móveis candidatos
- **Tratamento:** método champion produzido no Notebook 02; controle usa a regra atual documentada
- **Estratificação:** por `categoria_hsm`, DDD e faixa de quantidade de telefones quando operacionalmente possível
- **Tamanho amostral:** calculado sobre entrega por CPF elegível via diferença de duas proporções (Cohen's h)
- **Teste estatístico:** z-test para duas proporções
- **Duração:** 1-2 semanas completas (para capturar efeitos de dia da semana)

---

## Principais Premissas

1. **Status `processing` excluído** — status intermediário sem resultado final
2. **Telefones fixos excluídos** — WhatsApp não entrega em linhas fixas
3. **Entrega como métrica primária operacional/experimental** — mede se o telefone está quente; leitura é acompanhada como métrica secundária de negócio
4. **Wilson Score para ranqueamento** — mais robusto que taxa bruta para volumes desbalanceados
5. **Regressão Logística para calibração** — pesos baseados em dados para o score do modelo; heurística com pesos arbitrários como baseline comparativo
6. **Randomização por CPF** — evita contaminação entre grupos no A/B test
7. **Join causal** — uma fonte só recebe crédito se o telefone já existia naquela fonte antes do envio

---

## Limitações Conhecidas

1. **Schema desatualizado:** o `whatsapp_schema.yml` possui divergências em relação aos dados reais (colunas inexistentes, nomes diferentes). Documentado no Notebook 00.
2. **Regressão Logística:** assume linearidade no log-odds; não captura interações complexas. Modelos de árvore (gradient boosting) podem capturar interações não-lineares e serão explorados como extensão.
3. **Viés de seleção nos dados históricos:** os disparos já foram feitos com alguma lógica implícita; o modelo aprende padrões desse processo.
4. **Validação offline com holdout compartilhado:** o grid de half-life e a comparação entre métodos usam o mesmo set de validação, o que pode inflacionar estimativas. A decisão final fica para o A/B test.
5. **Custo por entrega:** não disponível na base, mas é uma métrica importante para o negócio.
6. **Heurística com pesos arbitrários:** os pesos não são derivados de calibração estatística; servem como baseline comparativo deliberado.
7. **Campo `validacao_telefone`** (dict com tipo de validação do telefone): disponível nos dados mas não incluído como feature. Pode conter sinais adicionais de qualidade para investigação futura.

---

## Tecnologias Utilizadas

- Python 3.12
- pandas, numpy, pyarrow
- matplotlib, seaborn
- scipy, statsmodels
- scikit-learn (Regressão Logística)
- Jupyter Notebooks
- pytest (testes unitários)

---

## Autor

Matheus de Andrade Santos
