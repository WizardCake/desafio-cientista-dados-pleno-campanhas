# Desafio Técnico - Cientista de Dados Pleno - Squad WhatsApp

---

## Contexto

Na Prefeitura do Rio, enviamos milhares de mensagens por mês via WhatsApp. Cada disparo tem um custo e as janelas de comunicação com o cidadão são preciosas. 

O **Registro Municipal Integrado (RMI)** consolida dados de múltiplos sistemas (Saúde, Educação, Assistência Social, IPTU, etc.). Um desafio crítico é a **Multiplicidade**: um mesmo cidadão pode ter vários telefones vinculados a ele, muitas vezes antigos ou desatualizados. 

Como Cientista de Dados no Squad WhatsApp, seu objetivo é criar a **Inteligência de Escolha**: identificar quais fontes de dados são mais confiáveis ("quentes") para garantir que mensagens críticas cheguem ao cidadão de forma eficiente e com menor custo.

---

## Instruções

1. Faça um fork do repositório do desafio para colocar a sua solução
2. Use **Jupyter Notebooks** (.ipynb) bem documentados ou scripts Python/SQL
3. Inclua **README.md** explicando sua abordagem, premissas e como reproduzir
4. **Entrega**: Envie o link do repositório para `selecao.pcrj@gmail.com`

---

## Dados

Você terá acesso a duas tabelas principais mascaradas para garantir anonimato e consistência:

### 1. Tabela de Performance: `base_disparo_mascarado`
Histórico real de disparos efetuados pelo motor de mensagens.

### 2. Tabela de Dimensão: `dim_telefone_mascarado`
Conhecimento consolidado sobre os telefones e suas origens.


**⚠️ DISCLAIMER SOBRE VIESES**: Algumas bases de dados já são consideradas "mais quentes" pela Prefeitura e aparecem com maior frequência nos logs. Identifique se uma base performa melhor porque é realmente superior ou se os números estão inflados pelo volume de tentativas (viés de seleção).

### Acesso aos dados

Os arquivos Parquet estão disponíveis no bucket GCS:

```
https://console.cloud.google.com/storage/browser/case_vagas/whatsapp
```
---

## Parte 1: Análise Exploratória e Qualidade de Fontes

**O objetivo aqui é medir o "calor" de cada sistema de origem.**

### 1. Desestruturação e Correlação
Um telefone pode ter vindo de vários sistemas. Use seus conhecimentos para correlacionar cada sistema de origem (`id_sistema_mask`) com a performance real nos disparos (`status_disparo`).

**Entregue**: Análise comparativa de taxas de entrega (`DELIVERED`) agregadas por sistema de origem.

### 2. Janela de Atualidade
Investigue se o tempo decorrido desde a última atualização do telefone no sistema de origem (`registro_data_atualizacao`) impacta na chance de sucesso do disparo.

**Entregue**: Análise de "decaimento" da qualidade do dado ao longo do tempo. Existe um "prazo de validade" para um telefone ser considerado quente?

---

## Parte 2: Inteligência de Priorização

**O objetivo aqui é criar a regra de negócio que o motor de disparos seguirá.**

### 3. Ranking de Sistemas
Com base nas análises anteriores, crie um ranking de confiabilidade para os sistemas da Prefeitura. 

**Entregue**: Tabela ou Score de ranking das fontes. Explique matematicamente por que o sistema X é melhor que o sistema Y.

### 4. Algoritmo de Escolha

Imagine que você tem 3 telefones diferentes para o mesmo CPF. Proponha um algoritmo (score ou ranking) que escolha automaticamente os **dois melhores** para receberem a mensagem.

**Entregue**: Explicação da lógica do algoritmo. Como você combina a "origem do dado" com a "data de atualização" e o "DDD" para tomar essa decisão?

---

## Parte 3: Desenho de Experimento

### 5. Proposta de Teste A/B
Como você validaria que seu novo ranking é realmente melhor do que a estratégia de envio aleatório (ou baseada em ordem alfabética) que usamos hoje?

**Entregue**: Desenho do experimento. Defina hipótese nula, métricas primárias e secundárias, tamanho de amostra e tempo de duração estimado para o teste.

---

## Avaliação

Você será avaliado nos seguintes critérios:

- **Manipulação de Dados (SQL/Python)**: Capacidade de lidar com arrays e joins complexos.
- **Raciocínio Analítico e Estatístico**: Tratamento de vieses e solidez na definição de métricas.
- **Visão de Negócio e Impacto**: Tradução da análise em uma regra de negócio acionável.
- **Comunicação e Visualização**: Clareza na apresentação dos resultados.

---

## FAQ

**1. O que define um telefone "quente"?**
Aquele que tem maior probabilidade de estar ativo, ser entregue e lido pelo cidadão correto.

**2. Posso usar ferramentas de BI?**
Sim, mas o core da análise e a lógica do algoritmo devem estar documentados no repositório.

---

## Contato

Dúvidas? Mande um e-mail para `patricia.catandi@prefeitura.rio` com o título começando com `[CASE DS]` 

Boa sorte! 🚀