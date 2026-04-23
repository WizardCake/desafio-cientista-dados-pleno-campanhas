"""
utils.py
Funções utilitárias compartilhadas entre os notebooks do desafio.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint


# ============================================================
# CONFIGURAÇÃO DE PATHS
# ============================================================

# Resolve o diretório base independente de onde o script é importado
# utils.py está em src/, então subimos 2 níveis para a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'

PATH_DISPARO = DATA_DIR / 'whatsapp_base_disparo_mascarado.parquet'
PATH_TELEFONE = DATA_DIR / 'whatsapp_dim_telefone_mascarado.parquet'


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados():
    """
    Carrega os dois dataframes principais.
    
    Returns:
        tuple: (df_disparo, df_telefone)
    """
    df_disparo = pd.read_parquet(PATH_DISPARO)
    df_telefone = pd.read_parquet(PATH_TELEFONE)
    return df_disparo, df_telefone


# ============================================================
# RELATÓRIO DE MISSING VALUES
# ============================================================

def missing_report(df, nome):
    """
    Exibe relatório de missing values ordenado por % de nulos.
    """
    missing = df.isnull().sum()
    pct = 100 * missing / len(df)
    report = pd.DataFrame({
        'coluna': df.columns,
        'nulos': missing.values,
        'pct_nulos': pct.values
    }).sort_values('pct_nulos', ascending=False)
    print(f'=== MISSING VALUES: {nome} ===')
    display(report[report['nulos'] > 0])
    print()


# ============================================================
# FILTROS E PREPARAÇÃO
# ============================================================

def filtrar_status_invalidos(df_disparo):
    """
    Remove disparos com status 'processing' (status intermediário
    sem resultado final definido).
    
    Premissa: 'processing' não reflete sucesso ou falha, portanto
    não deve entrar no cálculo de taxas.
    """
    n_antes = len(df_disparo)
    df = df_disparo[df_disparo['status_disparo'] != 'processing'].copy()
    n_depois = len(df)
    print(f'Filtrando status=processing: {n_antes:,} -> {n_depois:,} (-{n_antes - n_depois:,})')
    return df


def filtrar_telefones_fixos(df_telefone):
    """
    Remove telefones fixos da análise, pois WhatsApp Business
    não entrega mensagens para linhas fixas.
    
    Premissa: números fixos têm taxa de entrega zero por restrição
    técnica da plataforma, não por qualidade da fonte de dados.
    """
    n_antes = len(df_telefone)
    df = df_telefone[df_telefone['telefone_tipo'] != 'Fixo'].copy()
    n_depois = len(df)
    print(f'Filtrando telefones fixos: {n_antes:,} -> {n_depois:,} (-{n_antes - n_depois:,})')
    return df


# ============================================================
# EXPLOSÃO DO ARRAY DE APARIÇÕES
# ============================================================

def explodir_aparicoes(df_telefone):
    """
    Explode o campo telefone_aparicoes (list de dicts) em um
    dataframe com uma linha por (telefone, sistema).
    
    Returns:
        DataFrame com colunas: telefone_numero, id_sistema, cpf,
        proprietario_tipo, registro_data_atualizacao
    """
    # Explodir o array
    df_exploded = df_telefone[['telefone_numero', 'telefone_aparicoes']].explode('telefone_aparicoes')
    
    # Normalizar o dict em colunas
    df_aparicoes = pd.json_normalize(df_exploded['telefone_aparicoes'])
    df_aparicoes['telefone_numero'] = df_exploded['telefone_numero'].values
    
    # Renomear para consistência
    df_aparicoes = df_aparicoes.rename(columns={
        'id_sistema': 'id_sistema',
        'cpf': 'cpf_sistema',
        'proprietario_tipo': 'proprietario_tipo',
        'registro_data_atualizacao': 'registro_data_atualizacao'
    })
    
    print(f'Telefones únicos: {df_exploded["telefone_numero"].nunique():,}')
    print(f'Linhas após explosão: {len(df_aparicoes):,}')
    print(f'Sistemas únicos: {df_aparicoes["id_sistema"].nunique()}')
    
    return df_aparicoes


# ============================================================
# JOIN DISPARO x SISTEMA
# ============================================================

def join_disparo_sistema(df_disparo, df_aparicoes):
    """
    Realiza o join entre base de disparos e aparições por sistema.
    
    Chave: contato_telefone (disparo) == telefone_numero (aparições)
    
    Atenção: cada disparo pode gerar múltiplas linhas se o telefone
    aparece em N sistemas. Isso é intencional para atribuir o resultado
    a cada fonte de origem.
    """
    n_disparos_antes = df_disparo['id_disparo'].nunique()
    
    df_merged = df_disparo.merge(
        df_aparicoes,
        left_on='contato_telefone',
        right_on='telefone_numero',
        how='inner'
    )
    
    n_disparos_depois = df_merged['id_disparo'].nunique()
    n_linhas = len(df_merged)
    
    print(f'Disparos com match: {n_disparos_depois:,} / {n_disparos_antes:,}')
    print(f'Total de linhas após join: {n_linhas:,}')
    print(f'Multiplicidade média: {n_linhas / n_disparos_depois:.2f}x')
    
    return df_merged


# ============================================================
# MÉTRICAS POR SISTEMA
# ============================================================

def calcular_metricas_sistema(df_disparo_sistema):
    """
    Calcula métricas agregadas por sistema de origem.
    
    Métricas primárias (negócio):
    - taxa_leitura: read / total (objetivo final = cidadão lê a mensagem)
    - taxa_entrega: (delivered + read) / total (proxy de número ativo)
    
    Métricas auxiliares:
    - taxa_falha: failed / total
    """
    metricas = df_disparo_sistema.groupby('id_sistema').agg(
        total_disparos=('id_disparo', 'count'),
        read=('status_disparo', lambda x: (x == 'read').sum()),
        delivered=('status_disparo', lambda x: (x == 'delivered').sum()),
        failed=('status_disparo', lambda x: (x == 'failed').sum()),
        sent=('status_disparo', lambda x: (x == 'sent').sum()),
    ).reset_index()
    
    # Taxas
    metricas['taxa_leitura'] = metricas['read'] / metricas['total_disparos']
    metricas['taxa_entrega'] = (metricas['delivered'] + metricas['read']) / metricas['total_disparos']
    metricas['taxa_falha'] = metricas['failed'] / metricas['total_disparos']
    
    # Ordenar por taxa de leitura (primária)
    metricas = metricas.sort_values('taxa_leitura', ascending=False)
    
    return metricas


# ============================================================
# WILSON SCORE INTERVAL (LIMITE INFERIOR)
# ============================================================

def wilson_lower_bound(successes, total, alpha=0.05):
    """
    Retorna o limite inferior do intervalo de confiança de Wilson.
    
    Ideal para ranquear proporções com volumes desbalanceados,
    pois penaliza sistemas com pouca evidência.
    
    Args:
        successes: número de sucessos
        total: número total de tentativas
        alpha: nível de significância (default 5%)
    
    Returns:
        float: limite inferior do IC (0 a 1)
    """
    if total == 0:
        return 0.0
    ci_low, _ = proportion_confint(successes, total, alpha=alpha, method='wilson')
    return ci_low


def aplicar_wilson(metricas, col_sucessos='read', col_total='total_disparos'):
    """
    Aplica o Wilson Lower Bound às métricas por sistema.
    """
    metricas['wilson_lower'] = metricas.apply(
        lambda row: wilson_lower_bound(row[col_sucessos], row[col_total]),
        axis=1
    )
    return metricas.sort_values('wilson_lower', ascending=False)


# ============================================================
# DECAIMENTO TEMPORAL
# ============================================================

def calcular_decaimento_temporal(df_disparo_sistema, bins=None, labels=None):
    """
    Calcula taxas de sucesso por faixa de idade do dado.
    
    Args:
        df_disparo_sistema: dataframe com coluna 'registro_data_atualizacao'
        bins: limites dos bins em dias (default: [0, 30, 90, 180, 365, 730, 9999])
        labels: rótulos dos bins
    
    Returns:
        DataFrame com taxas por faixa
    """
    if bins is None:
        bins = [0, 30, 90, 180, 365, 730, 9999]
    if labels is None:
        labels = ['<30d', '30-90d', '90-180d', '180d-1a', '1-2a', '>2a']
    
    # Calcular dias entre envio e atualização
    df = df_disparo_sistema.copy()
    df['dias_desde_atualizacao'] = (
        pd.to_datetime(df['envio_datahora']) - 
        pd.to_datetime(df['registro_data_atualizacao'])
    ).dt.days
    
    # Filtrar valores negativos (inconsistência de data)
    n_negativos = (df['dias_desde_atualizacao'] < 0).sum()
    if n_negativos > 0:
        print(f'Atenção: {n_negativos:,} registros com dias negativos (data de atualização > envio). Serão excluídos.')
        df = df[df['dias_desde_atualizacao'] >= 0]
    
    # Criar faixas
    df['faixa_atualizacao'] = pd.cut(
        df['dias_desde_atualizacao'],
        bins=bins,
        labels=labels
    )
    
    # Agregar por faixa
    decaimento = df.groupby('faixa_atualizacao').agg(
        total=('status_disparo', 'count'),
        read=('status_disparo', lambda x: (x == 'read').sum()),
        delivered=('status_disparo', lambda x: (x == 'delivered').sum()),
        failed=('status_disparo', lambda x: (x == 'failed').sum()),
    ).reset_index()
    
    decaimento['taxa_leitura'] = decaimento['read'] / decaimento['total']
    decaimento['taxa_entrega'] = (decaimento['delivered'] + decaimento['read']) / decaimento['total']
    
    return decaimento
