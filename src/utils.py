"""
Funcoes utilitarias compartilhadas entre os notebooks do desafio.

A metrica operacional primaria desta solucao e entrega:
status_disparo em {'delivered', 'read'}.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PATH_DISPARO = DATA_DIR / "whatsapp_base_disparo_mascarado.parquet"
PATH_TELEFONE = DATA_DIR / "whatsapp_dim_telefone_mascarado.parquet"

STATUS_ENTREGA = {"delivered", "read"}
STATUS_INVALIDOS = {"processing"}
TIPOS_FIXOS = {"fixo", "fixa"}


# ============================================================
# CARREGAMENTO E NORMALIZACAO
# ============================================================

def carregar_dados():
    """Carrega os dois dataframes principais."""
    return pd.read_parquet(PATH_DISPARO), pd.read_parquet(PATH_TELEFONE)


def normalizar_status(df_disparo):
    """Padroniza status para lowercase sem alterar o dataframe original."""
    df = df_disparo.copy()
    df["status_disparo"] = df["status_disparo"].astype("string").str.strip().str.lower()
    return df


def normalizar_tipo_telefone(df_telefone):
    """Cria telefone_tipo_norm para filtros robustos a casing/acentos simples."""
    df = df_telefone.copy()
    df["telefone_tipo_norm"] = df["telefone_tipo"].astype("string").str.strip().str.lower()
    return df


def missing_report(df, nome):
    """Exibe relatorio de missing values ordenado por percentual de nulos."""
    missing = df.isnull().sum()
    pct = 100 * missing / len(df)
    report = pd.DataFrame({
        "coluna": df.columns,
        "nulos": missing.values,
        "pct_nulos": pct.values,
    }).sort_values("pct_nulos", ascending=False)
    print(f"=== MISSING VALUES: {nome} ===")
    display(report[report["nulos"] > 0])
    print()


# ============================================================
# FILTROS
# ============================================================

def filtrar_status_invalidos(df_disparo):
    """
    Remove status intermediarios sem resultado final definido.
    """
    df_norm = normalizar_status(df_disparo)
    n_antes = len(df_norm)
    df = df_norm[~df_norm["status_disparo"].isin(STATUS_INVALIDOS)].copy()
    n_depois = len(df)
    print(f"Filtrando status intermediarios: {n_antes:,} -> {n_depois:,} (-{n_antes - n_depois:,})")
    return df


def filtrar_telefones_fixos(df_telefone):
    """
    Remove telefones fixos da analise.
    """
    df_norm = normalizar_tipo_telefone(df_telefone)
    n_antes = len(df_norm)
    df = df_norm[~df_norm["telefone_tipo_norm"].isin(TIPOS_FIXOS)].copy()
    n_depois = len(df)
    print(f"Filtrando telefones fixos: {n_antes:,} -> {n_depois:,} (-{n_antes - n_depois:,})")
    return df


# ============================================================
# APARICOES
# ============================================================

def explodir_aparicoes(df_telefone):
    """
    Explode telefone_aparicoes para uma linha por aparicao bruta.
    """
    df_exploded = df_telefone[["telefone_numero", "telefone_aparicoes"]].explode("telefone_aparicoes")
    df_aparicoes = pd.json_normalize(df_exploded["telefone_aparicoes"])
    df_aparicoes["telefone_numero"] = df_exploded["telefone_numero"].values

    df_aparicoes = df_aparicoes.rename(columns={"cpf": "cpf_sistema"})
    expected = ["telefone_numero", "id_sistema", "cpf_sistema", "proprietario_tipo", "registro_data_atualizacao"]
    for col in expected:
        if col not in df_aparicoes.columns:
            df_aparicoes[col] = np.nan

    df_aparicoes = df_aparicoes[expected].copy()
    df_aparicoes["registro_data_atualizacao"] = pd.to_datetime(
        df_aparicoes["registro_data_atualizacao"],
        errors="coerce",
    )

    print(f"Telefones unicos: {df_aparicoes['telefone_numero'].nunique():,}")
    print(f"Aparicoes brutas: {len(df_aparicoes):,}")
    print(f"Sistemas unicos: {df_aparicoes['id_sistema'].nunique():,}")
    return df_aparicoes


def preparar_aparicoes_por_fonte(df_aparicoes):
    """
    Uma linha por (telefone_numero, id_sistema).

    Mantem a data de atualizacao mais recente para evitar que um telefone
    com muitas aparicoes no mesmo sistema infle o ranking da fonte.
    """
    df = df_aparicoes.dropna(subset=["telefone_numero", "id_sistema"]).copy()
    df = (
        df.sort_values("registro_data_atualizacao")
        .groupby(["telefone_numero", "id_sistema"], as_index=False)
        .agg(
            registro_data_atualizacao=("registro_data_atualizacao", "max"),
            cpfs_sistema_distintos=("cpf_sistema", lambda s: pd.Series(s).dropna().nunique()),
            qtd_aparicoes_brutas=("cpf_sistema", "size"),
        )
    )
    print(f"Aparicoes por fonte deduplicadas: {len(df):,}")
    return df


def preparar_telefone_cpf(df_aparicoes):
    """Uma linha por par (telefone_numero, cpf_sistema) para gerar candidatos por CPF."""
    df = (
        df_aparicoes[["telefone_numero", "cpf_sistema"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"cpf_sistema": "cpf"})
        .copy()
    )
    print(f"Pares telefone-CPF candidatos: {len(df):,}")
    return df


def preparar_metadados_telefone(df_telefone, df_aparicoes_fonte):
    """Monta metadados estaticos de telefone usados no score operacional."""
    meta = df_telefone[[
        "telefone_numero",
        "telefone_ddd",
        "telefone_proprietarios_quantidade",
        "telefone_sistemas_quantidade",
        "telefone_qualidade",
    ]].copy()
    meta = meta.rename(columns={
        "telefone_proprietarios_quantidade": "n_proprietarios",
        "telefone_sistemas_quantidade": "n_sistemas_telefone_dim",
    })
    meta["n_proprietarios"] = meta["n_proprietarios"].fillna(1).clip(lower=1)
    meta["penalidade_proprietarios"] = 1 / meta["n_proprietarios"]

    n_sistemas = (
        df_aparicoes_fonte.groupby("telefone_numero")["id_sistema"]
        .nunique()
        .reset_index(name="n_sistemas_telefone")
    )
    meta = meta.merge(n_sistemas, on="telefone_numero", how="left")
    meta["n_sistemas_telefone"] = meta["n_sistemas_telefone"].fillna(0).astype(int)
    return meta


# ============================================================
# JOIN E METRICAS
# ============================================================

def join_disparo_sistema(df_disparo, df_aparicoes_fonte, causal=False):
    """
    Join entre disparos e fontes deduplicadas por telefone.

    Se causal=True, mantem apenas aparicoes cuja data ja existia no momento
    do disparo. Aparicoes sem data tambem ficam fora do ranking causal.
    """
    n_disparos_antes = df_disparo["id_disparo"].nunique()
    df = df_disparo.merge(
        df_aparicoes_fonte,
        left_on="contato_telefone",
        right_on="telefone_numero",
        how="inner",
    )

    if causal:
        envio = pd.to_datetime(df["envio_datahora"])
        atualizacao = pd.to_datetime(df["registro_data_atualizacao"])
        df = df[atualizacao.notna() & (atualizacao <= envio)].copy()

    n_disparos_depois = df["id_disparo"].nunique()
    n_linhas = len(df)
    multiplicidade = n_linhas / n_disparos_depois if n_disparos_depois else 0

    print(f"Disparos com match: {n_disparos_depois:,} / {n_disparos_antes:,}")
    print(f"Total de linhas apos join: {n_linhas:,}")
    print(f"Multiplicidade media: {multiplicidade:.2f}x")
    return df


def _contar_status_unico(grupo, status):
    return grupo.loc[grupo["status_disparo"].eq(status), "id_disparo"].nunique()


def calcular_metricas_sistema(df_disparo_sistema):
    """
    Calcula metricas por sistema com denominador em disparos unicos.
    """
    rows = []
    for id_sistema, grupo in df_disparo_sistema.groupby("id_sistema", dropna=False):
        total = grupo["id_disparo"].nunique()
        read = _contar_status_unico(grupo, "read")
        delivered = _contar_status_unico(grupo, "delivered")
        failed = _contar_status_unico(grupo, "failed")
        sent = _contar_status_unico(grupo, "sent")
        rows.append({
            "id_sistema": id_sistema,
            "total_disparos": total,
            "read": read,
            "delivered": delivered,
            "failed": failed,
            "sent": sent,
        })

    metricas = pd.DataFrame(rows)
    if metricas.empty:
        return pd.DataFrame(columns=[
            "id_sistema", "total_disparos", "read", "delivered", "failed", "sent",
            "sucessos_entrega", "taxa_entrega", "taxa_leitura", "taxa_falha",
        ])

    metricas["sucessos_entrega"] = metricas["read"] + metricas["delivered"]
    metricas["taxa_entrega"] = metricas["sucessos_entrega"] / metricas["total_disparos"]
    metricas["taxa_leitura"] = metricas["read"] / metricas["total_disparos"]
    metricas["taxa_falha"] = metricas["failed"] / metricas["total_disparos"]
    return metricas.sort_values("taxa_entrega", ascending=False).reset_index(drop=True)


def wilson_lower_bound(successes, total, alpha=0.05):
    """Limite inferior do intervalo de confianca de Wilson."""
    if total == 0:
        return 0.0
    ci_low, _ = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return ci_low


def aplicar_wilson(metricas, col_sucessos="sucessos_entrega", col_total="total_disparos", nome_coluna="wilson_lower_entrega"):
    """Aplica Wilson Lower Bound e ordena de forma conservadora."""
    df = metricas.copy()
    df[nome_coluna] = df.apply(
        lambda row: wilson_lower_bound(row[col_sucessos], row[col_total]),
        axis=1,
    )
    return df.sort_values(nome_coluna, ascending=False).reset_index(drop=True)


def normalizar_0_1(series):
    """Normalizacao min-max robusta para series constantes."""
    series = series.astype(float)
    if series.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def calcular_score_sistema(metricas):
    """Score conservador de fonte, alinhado a entrega."""
    df = aplicar_wilson(metricas)
    df["score_sistema"] = normalizar_0_1(df["wilson_lower_entrega"])
    return df.sort_values("score_sistema", ascending=False).reset_index(drop=True)


# ============================================================
# TEMPO E PRIORS
# ============================================================

def adicionar_features_temporais(df, half_life, reference_col="envio_datahora"):
    """
    Adiciona recencia causal e decaimento temporal.
    """
    out = df.copy()
    ref = pd.to_datetime(out[reference_col])
    atualizacao = pd.to_datetime(out["registro_data_atualizacao"])
    dias = (ref - atualizacao).dt.days
    out["tem_data_causal"] = atualizacao.notna() & dias.ge(0)
    out["dias_desde_atualizacao"] = np.where(out["tem_data_causal"], dias, 9999)
    out["decaimento_temporal"] = np.exp(-np.log(2) * out["dias_desde_atualizacao"] / half_life)
    return out


def calcular_decaimento_temporal(df_disparo_sistema, bins=None, labels=None):
    """Calcula taxas por faixa de idade do dado usando eventos causais."""
    if bins is None:
        bins = [0, 30, 90, 180, 365, 730, 9999]
    if labels is None:
        labels = ["<30d", "30-90d", "90-180d", "180d-1a", "1-2a", ">2a"]

    df = adicionar_features_temporais(df_disparo_sistema, half_life=90)
    n_nao_causais = (~df["tem_data_causal"]).sum()
    if n_nao_causais > 0:
        print(f"Registros nao causais ou sem data excluidos do decaimento: {n_nao_causais:,}")
        df = df[df["tem_data_causal"]].copy()

    df["faixa_atualizacao"] = pd.cut(df["dias_desde_atualizacao"], bins=bins, labels=labels)
    rows = []
    for faixa, grupo in df.groupby("faixa_atualizacao", observed=False):
        total = grupo["id_disparo"].nunique()
        read = grupo.loc[grupo["status_disparo"].eq("read"), "id_disparo"].nunique()
        delivered = grupo.loc[grupo["status_disparo"].eq("delivered"), "id_disparo"].nunique()
        failed = grupo.loc[grupo["status_disparo"].eq("failed"), "id_disparo"].nunique()
        rows.append({
            "faixa_atualizacao": faixa,
            "total": total,
            "read": read,
            "delivered": delivered,
            "failed": failed,
        })
    decaimento = pd.DataFrame(rows)

    decaimento["sucessos_entrega"] = decaimento["read"] + decaimento["delivered"]
    decaimento["taxa_entrega"] = decaimento["sucessos_entrega"] / decaimento["total"]
    decaimento["taxa_leitura"] = decaimento["read"] / decaimento["total"]
    return decaimento


def calcular_prior_suavizado(df, grupo_col, target_col, alpha=50):
    """Prior bayesiano simples para grupos como DDD."""
    baseline = df[target_col].mean()
    prior = df.groupby(grupo_col)[target_col].agg(["sum", "count"]).reset_index()
    prior["score_prior"] = (prior["sum"] + alpha * baseline) / (prior["count"] + alpha)
    return prior[[grupo_col, "score_prior"]], baseline


def preparar_metricas_cpf(df_disparo):
    """Agrega resultado historico no nivel CPF para desenho experimental."""
    df = normalizar_status(df_disparo).dropna(subset=["cpf"]).copy()
    df["y_entrega"] = df["status_disparo"].isin(STATUS_ENTREGA).astype(int)
    df["y_read"] = df["status_disparo"].eq("read").astype(int)
    df["y_failed"] = df["status_disparo"].eq("failed").astype(int)
    return df.groupby("cpf").agg(
        total_disparos=("id_disparo", "nunique"),
        cpf_teve_read=("y_read", "max"),
        cpf_teve_entrega=("y_entrega", "max"),
        cpf_teve_falha=("y_failed", "max"),
        primeira_data=("envio_datahora", "min"),
        ultima_data=("envio_datahora", "max"),
    ).reset_index()
