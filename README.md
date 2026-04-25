# Desafio Técnico — Cientista de Dados Pleno (Squad WhatsApp)

## Objetivo

Este repositório contém a solução para o desafio técnico do Squad WhatsApp: construir uma **Inteligência de Escolha** para selecionar, para cada CPF com múltiplos telefones candidatos, os **dois melhores telefones** para receber mensagens via WhatsApp.

A solução responde às três partes do enunciado:

1. **Análise exploratória e qualidade de fontes:** medir o "calor" dos sistemas de origem dos telefones.
2. **Inteligência de priorização:** transformar esses sinais em uma regra operacional de escolha top-2 por CPF.
3. **Desenho de experimento:** propor um teste A/B para validar se a nova regra melhora a estratégia atual.

## Leitura recomendada

O principal documento de leitura é o [RELATORIO.md](RELATORIO.md). Ele consolida os quatro notebooks em uma narrativa única, explicando a progressão lógica da solução:

```text
inspeção dos dados
=> definição da métrica
=> join causal entre disparos e fontes
=> ranking de sistemas
=> decaimento temporal
=> algoritmo de escolha top-2
=> validação offline
=> desenho de teste A/B
```

O relatório é a melhor porta de entrada para o leitor entender o nexo entre as etapas e por que cada decisão metodológica foi tomada. Já os notebooks são a implementação analítica detalhada, onde cada passo e seus resultados estão registrados.

## Estrutura do repositório

```text
desafio-cientista-dados-pleno-campanhas/
├── README.md
├── RELATORIO.md
├── requirements.txt
├── enunciado/
│   └── README.md
├── notebooks/
│   ├── 00_inspecao_inicial.ipynb
│   ├── 01_eda_e_qualidade_fontes.ipynb
│   ├── 02_inteligencia_priorizacao.ipynb
│   └── 03_desenho_experimento.ipynb
├── src/
│   └── utils.py
├── tests/
│   └── test_utils.py
└── data/
    ├── raw/
    └── processed/
```

## Notebooks

Os notebooks devem ser lidos e executados na ordem abaixo.

| Notebook | Papel na solução |
|---|---|
| `00_inspecao_inicial.ipynb` | Valida schema, chaves de join, status, cobertura e premissas iniciais. |
| `01_eda_e_qualidade_fontes.ipynb` | Mede a qualidade das fontes com join causal, ranking Empirical Bayes e análise de decaimento temporal. |
| `02_inteligencia_priorizacao.ipynb` | Constrói a regra de escolha top-2 por CPF, compara modelo, heurística e baselines, e define o champion. |
| `03_desenho_experimento.ipynb` | Desenha o teste A/B, define hipóteses, métricas, guardrails, tamanho amostral e duração estimada. |

## Abordagem resumida

A métrica operacional principal é **entrega**, definida como:

```text
status_disparo em {delivered, read}
```

O status `read` é contabilizado como entrega porque representa um estado posterior ao recebimento: se a mensagem foi lida, ela necessariamente chegou ao WhatsApp do destinatário. A leitura também é acompanhada separadamente como métrica secundária de negócio.

A análise usa um **join causal** entre disparos e fontes. Uma fonte só recebe crédito por um disparo se o telefone já existia naquela fonte antes do envio:

```text
registro_data_atualizacao <= envio_datahora
```

Esse cuidado evita vazamento temporal e reduz o risco de inflar uma fonte que só registrou o telefone depois do disparo.

O ranking dos sistemas usa **Empirical Bayes Lower Bound**, que combina taxa observada e incerteza amostral. Isso evita promover fontes pequenas apenas por flutuação de poucos disparos bem-sucedidos.

Na etapa de priorização, foram comparados modelo logístico, heurística ponderada e baselines determinísticos. A política candidata final foi `fonte_mais_recente`, por ter melhor comportamento na validação offline de decisão e por ser simples de explicar operacionalmente. Ela é tratada como **champion candidato para A/B**, não como prova definitiva de ganho.

## Premissas principais

- `processing` é status intermediário e não entra no cálculo de desfecho.
- Telefones fixos são excluídos por premissa operacional de WhatsApp.
- Entrega operacional é `delivered` ou `read`.
- O join entre disparos e fontes é causal.
- O ranking de sistemas é conservador, usando Empirical Bayes.
- A validação offline é indicativa, não causal.
- A decisão final de substituição da regra deve ser feita por teste A/B randomizado por CPF.

## Como reproduzir

### 1. Criar ambiente e instalar dependências

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Em Linux/Mac:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Obter os dados

Os dados do desafio estão no bucket indicado no enunciado:

```text
https://console.cloud.google.com/storage/browser/case_vagas/whatsapp
```

O notebook `00_inspecao_inicial.ipynb` contém a lógica de verificação dos arquivos locais e download quando necessário.

Arquivos esperados em `data/raw/`:

```text
whatsapp_base_disparo_mascarado.parquet
whatsapp_dim_telefone_mascarado.parquet
whatsapp_schema.yml
```

### 3. Executar os notebooks

Execute na ordem:

```text
notebooks/00_inspecao_inicial.ipynb
notebooks/01_eda_e_qualidade_fontes.ipynb
notebooks/02_inteligencia_priorizacao.ipynb
notebooks/03_desenho_experimento.ipynb
```

Os artefatos intermediários são gravados em `data/processed/`.

### 4. Rodar testes unitários

```bash
python -m pytest tests/test_utils.py -v
```

## Arquivos de apoio

- `src/utils.py`: funções compartilhadas pelos notebooks, incluindo filtros, joins causais, métricas, ranking, priorização e avaliação offline.
- `tests/test_utils.py`: testes unitários das principais funções utilitárias.
- `RELATORIO.md`: relatório consolidado, com a explicação narrativa completa da solução.
- `enunciado/README.md`: enunciado original do desafio.
