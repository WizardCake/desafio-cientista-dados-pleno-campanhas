"""
Funções utilitárias compartilhadas entre os notebooks do desafio.

A métrica operacional primária desta solução é entrega:
status_disparo em {'delivered', 'read'}.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm


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

QUALIDADE_MAP = {
    "ALTA": 1.0,
    "alta": 1.0,
    "MEDIA": 0.5,
    "media": 0.5,
    "MÉDIA": 0.5,
    "média": 0.5,
    "BAIXA": 0.0,
    "baixa": 0.0,
}
QUALIDADE_DEFAULT = 0.25

SEED = 42
RANDOM_SEEDS = list(range(20))
HALF_LIFE_GRID = [30, 60, 90, 120, 180, 270, 365]

FEATURE_COLS = [
    "max_score_origem_tempo",
    "melhor_score_sistema",
    "melhor_decaimento",
    "score_fonte_mais_recente",
    "decaimento_fonte_mais_recente",
    "decaimento_medio",
    "penalidade_proprietarios",
    "score_ddd",
    "is_ddd_21",
    "score_qualidade",
    "score_exclusividade_cpf",
    "log_cpfs_distintos_telefone",
    "log_n_sistemas_telefone",
    "proporcao_aparicoes_causais",
]

PESOS_HEURISTICA = {
    "max_score_origem_tempo": 0.35,
    "score_fonte_mais_recente": 0.10,
    "score_ddd": 0.15,
    "penalidade_proprietarios": 0.10,
    "score_exclusividade_cpf": 0.10,
    "score_qualidade": 0.08,
    "melhor_decaimento": 0.05,
    "log_n_sistemas_telefone": 0.04,
    "proporcao_aparicoes_causais": 0.03,
}

PESOS_HEURISTICA_NOTA = (
    "Pesos heurísticos escolhidos arbitrariamente como baseline comparativo. "
    "Não são derivados de calibração estatística — servem como referência "
    "contra a qual o modelo logístico é comparado no holdout."
)


# ============================================================
# CARREGAMENTO E NORMALIZAÇÃO
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
    """Exibe relatório de missing values ordenado por percentual de nulos."""
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
    """Remove status intermediários sem resultado final definido."""
    df_norm = normalizar_status(df_disparo)
    n_antes = len(df_norm)
    df = df_norm[~df_norm["status_disparo"].isin(STATUS_INVALIDOS)].copy()
    n_depois = len(df)
    print(f"Filtrando status intermediários: {n_antes:,} -> {n_depois:,} (-{n_antes - n_depois:,})")
    return df


def filtrar_telefones_fixos(df_telefone):
    """Remove telefones fixos da análise."""
    df_norm = normalizar_tipo_telefone(df_telefone)
    n_antes = len(df_norm)
    df = df_norm[~df_norm["telefone_tipo_norm"].isin(TIPOS_FIXOS)].copy()
    n_depois = len(df)
    print(f"Filtrando telefones fixos: {n_antes:,} -> {n_depois:,} (-{n_antes - n_depois:,})")
    return df


# ============================================================
# APARIÇÕES
# ============================================================

def explodir_aparicoes(df_telefone):
    """Explode telefone_aparicoes para uma linha por aparição bruta."""
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

    print(f"Telefones únicos: {df_aparicoes['telefone_numero'].nunique():,}")
    print(f"Aparições brutas: {len(df_aparicoes):,}")
    print(f"Sistemas únicos: {df_aparicoes['id_sistema'].nunique():,}")
    return df_aparicoes


def preparar_aparicoes_por_fonte(df_aparicoes):
    """
    Uma linha por (telefone_numero, id_sistema, registro_data_atualizacao).

    O join causal escolhe a ultima aparicao valida antes de cada envio.

    A deduplicacao aqui evita repeticoes brutas sem perder a linha do tempo.
    """
    df = df_aparicoes.dropna(subset=["telefone_numero", "id_sistema"]).copy()
    df["registro_data_atualizacao"] = pd.to_datetime(df["registro_data_atualizacao"], errors="coerce")
    df = (
        df.sort_values(["telefone_numero", "id_sistema", "registro_data_atualizacao"])
        .groupby(["telefone_numero", "id_sistema", "registro_data_atualizacao"], as_index=False, dropna=False)
        .agg(
            cpfs_sistema_distintos=("cpf_sistema", lambda s: pd.Series(s).dropna().nunique()),
            qtd_aparicoes_brutas=("cpf_sistema", "size"),
        )
    )
    print(f"Aparições por fonte deduplicadas: {len(df):,}")
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


def preparar_metadados_telefone(df_telefone, df_aparicoes_fonte, df_aparicoes_brutas=None):
    """Monta metadados estáticos de telefone usados no score operacional, incluindo qualidade."""
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
    meta["telefone_ddd"] = pd.to_numeric(meta["telefone_ddd"], errors="coerce")
    meta["is_ddd_21"] = meta["telefone_ddd"].eq(21).astype(int)
    meta["n_proprietarios"] = meta["n_proprietarios"].fillna(1).clip(lower=1)
    meta["penalidade_proprietarios"] = 1 / meta["n_proprietarios"]

    meta["score_qualidade"] = (
        meta["telefone_qualidade"]
        .astype("string")
        .str.strip()
        .str.upper()
        .map(QUALIDADE_MAP)
        .fillna(QUALIDADE_DEFAULT)
    )

    n_sistemas = (
        df_aparicoes_fonte.groupby("telefone_numero")["id_sistema"]
        .nunique()
        .reset_index(name="n_sistemas_telefone")
    )
    meta = meta.merge(n_sistemas, on="telefone_numero", how="left")
    meta["n_sistemas_telefone"] = meta["n_sistemas_telefone"].fillna(0).astype(int)

    if df_aparicoes_brutas is not None and "cpf_sistema" in df_aparicoes_brutas.columns:
        cpfs_distintos = (
            df_aparicoes_brutas.groupby("telefone_numero")["cpf_sistema"]
            .nunique(dropna=True)
            .reset_index(name="cpfs_distintos_telefone")
        )
    elif "cpfs_sistema_distintos" in df_aparicoes_fonte.columns:
        cpfs_distintos = (
            df_aparicoes_fonte.groupby("telefone_numero")["cpfs_sistema_distintos"]
            .sum()
            .reset_index(name="cpfs_distintos_telefone")
        )
    else:
        cpfs_distintos = pd.DataFrame({
            "telefone_numero": meta["telefone_numero"],
            "cpfs_distintos_telefone": 1,
        })

    meta = meta.merge(cpfs_distintos, on="telefone_numero", how="left")
    meta["cpfs_distintos_telefone"] = meta["cpfs_distintos_telefone"].fillna(1).clip(lower=1)
    meta["score_exclusividade_cpf"] = 1 / meta["cpfs_distintos_telefone"]
    meta["log_cpfs_distintos_telefone"] = np.log1p(meta["cpfs_distintos_telefone"])
    return meta


# ============================================================
# JOIN E MÉTRICAS
# ============================================================

def _selecionar_aparicoes_evento_fonte(df_joinado, causal_only=True):
    """
    Seleciona no maximo uma aparicao por (disparo, sistema).

    Quando ha varias atualizacoes do mesmo telefone no mesmo sistema, mantem a
    ultima aparicao causal antes do envio. Se causal_only=False e uma fonte
    ainda nao tem aparicao causal naquele envio, preserva uma linha nao causal
    com score zerado para calculo de cobertura/proporcao causal.
    """
    keys = ["id_disparo", "id_sistema"]
    sort_cols = keys + ["registro_data_atualizacao"]

    if causal_only:
        base = df_joinado[df_joinado["tem_data_causal"]].copy()
        if base.empty:
            return base
        return (
            base.sort_values(sort_cols, na_position="first")
            .drop_duplicates(keys, keep="last")
            .reset_index(drop=True)
        )

    causal = (
        df_joinado[df_joinado["tem_data_causal"]]
        .sort_values(sort_cols, na_position="first")
        .drop_duplicates(keys, keep="last")
    )
    causal_keys = causal[keys].drop_duplicates()
    sem_causal = df_joinado.merge(
        causal_keys.assign(_tem_aparicao_causal=1),
        on=keys,
        how="left",
    )
    sem_causal = sem_causal[sem_causal["_tem_aparicao_causal"].isna()].drop(
        columns=["_tem_aparicao_causal"]
    )
    nao_causal = (
        sem_causal.sort_values(sort_cols, na_position="first")
        .drop_duplicates(keys, keep="last")
    )
    return pd.concat([causal, nao_causal], ignore_index=True)


def join_disparo_sistema(df_disparo, df_aparicoes_fonte, causal=False):
    """
    Join entre disparos e fontes deduplicadas por telefone.

    Se causal=True, mantém apenas aparições cuja data já existia no momento
    do disparo. Aparições sem data também ficam fora do ranking causal.
    """
    n_disparos_antes = df_disparo["id_disparo"].nunique()
    df = df_disparo.merge(
        df_aparicoes_fonte,
        left_on="contato_telefone",
        right_on="telefone_numero",
        how="inner",
    )

    df = adicionar_features_temporais(df, half_life=90, reference_col="envio_datahora")
    df = _selecionar_aparicoes_evento_fonte(df, causal_only=causal)

    n_disparos_depois = df["id_disparo"].nunique()
    n_linhas = len(df)
    multiplicidade = n_linhas / n_disparos_depois if n_disparos_depois else 0

    print(f"Disparos com match: {n_disparos_depois:,} / {n_disparos_antes:,}")
    print(f"Total de linhas após join: {n_linhas:,}")
    print(f"Multiplicidade média: {multiplicidade:.2f}x")
    return df


def calcular_metricas_sistema(df_disparo_sistema, metodo_atribuicao="full"):
    """
    Calcula metricas por sistema.

    metodo_atribuicao:
    - full: cada sistema causal associado ao telefone recebe credito integral.
    - fracionario: o credito do disparo e dividido entre sistemas causais.
    - fonte_mais_recente: apenas a fonte causal mais recente recebe credito.
    """
    if df_disparo_sistema.empty:
        return pd.DataFrame(columns=[
            "id_sistema", "total_disparos", "read", "delivered", "failed", "sent",
            "sucessos_entrega", "taxa_entrega", "taxa_leitura", "taxa_falha",
            "metodo_atribuicao",
        ])

    df = df_disparo_sistema.copy()
    metodo_atribuicao = metodo_atribuicao.lower()

    if metodo_atribuicao == "fonte_mais_recente":
        df = (
            df.sort_values(["id_disparo", "registro_data_atualizacao", "id_sistema"], na_position="first")
            .drop_duplicates("id_disparo", keep="last")
            .copy()
        )
        df["_peso_atribuicao"] = 1.0
    elif metodo_atribuicao == "fracionario":
        n_fontes = df.groupby("id_disparo")["id_sistema"].transform("nunique").clip(lower=1)
        df["_peso_atribuicao"] = 1 / n_fontes
    elif metodo_atribuicao == "full":
        df = df.drop_duplicates(["id_disparo", "id_sistema"]).copy()
        df["_peso_atribuicao"] = 1.0
    else:
        raise ValueError("metodo_atribuicao deve ser 'full', 'fracionario' ou 'fonte_mais_recente'.")

    df["_read"] = df["status_disparo"].eq("read").astype(float) * df["_peso_atribuicao"]
    df["_delivered"] = df["status_disparo"].eq("delivered").astype(float) * df["_peso_atribuicao"]
    df["_failed"] = df["status_disparo"].eq("failed").astype(float) * df["_peso_atribuicao"]
    df["_sent"] = df["status_disparo"].eq("sent").astype(float) * df["_peso_atribuicao"]

    metricas = df.groupby("id_sistema", dropna=False).agg(
        total_disparos=("_peso_atribuicao", "sum"),
        read=("_read", "sum"),
        delivered=("_delivered", "sum"),
        failed=("_failed", "sum"),
        sent=("_sent", "sum"),
    ).reset_index()

    metricas["sucessos_entrega"] = metricas["read"] + metricas["delivered"]
    metricas["taxa_entrega"] = metricas["sucessos_entrega"] / metricas["total_disparos"]
    metricas["taxa_leitura"] = metricas["read"] / metricas["total_disparos"]
    metricas["taxa_falha"] = metricas["failed"] / metricas["total_disparos"]
    metricas["metodo_atribuicao"] = metodo_atribuicao
    return metricas.sort_values("taxa_entrega", ascending=False).reset_index(drop=True)


def wilson_lower_bound(successes, total, alpha=0.05):
    """Limite inferior do intervalo de confiança de Wilson."""
    if total == 0:
        return 0.0
    z = norm.ppf(1 - alpha / 2)
    phat = successes / total
    denom = 1 + z**2 / total
    center = phat + z**2 / (2 * total)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return max((center - margin) / denom, 0.0)


def empirical_bayes_lower_bound(successes, total, global_successes, global_total, prior_strength=200, alpha=0.05):
    """
    Limite inferior beta-binomial com shrinkage para a media global.

    E mais adequado que Wilson quando ha poucos grupos, volumes desiguais e
    atribuicao fracionaria, porque o prior explicita o recuo para a taxa media
    observada em vez de tratar cada fonte como um binomio isolado.
    """
    if global_total <= 0:
        return 0.0
    global_rate = np.clip(global_successes / global_total, 1e-6, 1 - 1e-6)
    prior_alpha = global_rate * prior_strength
    prior_beta = (1 - global_rate) * prior_strength
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + max(total - successes, 0)
    return float(beta_dist.ppf(alpha / 2, posterior_alpha, posterior_beta))


def aplicar_empirical_bayes(metricas, col_sucessos="sucessos_entrega", col_total="total_disparos",
                            nome_coluna="eb_lower_entrega", prior_strength=200, alpha=0.05):
    """Aplica ranking beta-binomial empirico por fonte."""
    df = metricas.copy()
    global_successes = df[col_sucessos].sum()
    global_total = df[col_total].sum()
    global_rate = global_successes / global_total if global_total > 0 else 0
    df["posterior_mean_entrega"] = (
        df[col_sucessos] + global_rate * prior_strength
    ) / (df[col_total] + prior_strength)
    df[nome_coluna] = df.apply(
        lambda row: empirical_bayes_lower_bound(
            row[col_sucessos],
            row[col_total],
            global_successes,
            global_total,
            prior_strength=prior_strength,
            alpha=alpha,
        ),
        axis=1,
    )
    return df.sort_values(nome_coluna, ascending=False).reset_index(drop=True)


def aplicar_wilson(metricas, col_sucessos="sucessos_entrega", col_total="total_disparos", nome_coluna="wilson_lower_entrega"):
    """Aplica Wilson Lower Bound e ordena de forma conservadora."""
    df = metricas.copy()
    df[nome_coluna] = df.apply(
        lambda row: wilson_lower_bound(row[col_sucessos], row[col_total]),
        axis=1,
    )
    return df.sort_values(nome_coluna, ascending=False).reset_index(drop=True)


def normalizar_0_1(series):
    """Normalização min-max robusta para séries constantes."""
    series = series.astype(float)
    if series.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def calcular_score_sistema(metricas, metodo_ranking="empirical_bayes", prior_strength=200):
    """Score conservador de fonte, alinhado a entrega."""
    if metricas.empty:
        return metricas.copy()

    df = aplicar_wilson(metricas)
    df = aplicar_empirical_bayes(df, prior_strength=prior_strength)

    if metodo_ranking == "wilson":
        score_col = "wilson_lower_entrega"
    elif metodo_ranking in {"empirical_bayes", "eb_lower"}:
        score_col = "eb_lower_entrega"
    elif metodo_ranking == "posterior_mean":
        score_col = "posterior_mean_entrega"
    else:
        raise ValueError("metodo_ranking deve ser 'empirical_bayes', 'wilson' ou 'posterior_mean'.")

    df["score_sistema_raw"] = df[score_col]
    df["score_sistema"] = normalizar_0_1(df["score_sistema_raw"])
    df["metodo_ranking"] = metodo_ranking
    return df.sort_values("score_sistema", ascending=False).reset_index(drop=True)


# ============================================================
# TEMPO E PRIORS
# ============================================================

def adicionar_features_temporais(df, half_life, reference_col="envio_datahora"):
    """Adiciona recência causal e decaimento temporal."""
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
        print(f"Registros não causais ou sem data excluídos do decaimento: {n_nao_causais:,}")
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
    """Agrega resultado histórico no nível CPF para desenho experimental."""
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


# ============================================================
# PRIORIZAÇÃO — FUNÇÕES DO NOTEBOOK 02
# ============================================================

def criar_splits_temporais(df, time_col="envio_datahora", frac_treino=0.60, frac_tuning=0.20):
    """
    Cria splits temporais treino/tuning/teste por volume de eventos.

    O tuning escolhe hiperparametros como half-life. O teste fica isolado
    para avaliacao final offline.
    """
    if frac_treino <= 0 or frac_tuning <= 0 or frac_treino + frac_tuning >= 1:
        raise ValueError("Use frac_treino > 0, frac_tuning > 0 e soma menor que 1.")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    if n < 3:
        raise ValueError("Sao necessarios ao menos 3 eventos para split temporal.")

    idx_tuning = max(1, min(n - 2, int(n * frac_treino)))
    idx_teste = max(idx_tuning + 1, min(n - 1, int(n * (frac_treino + frac_tuning))))
    cutoff_tuning = df_sorted.loc[idx_tuning, time_col]
    cutoff_teste = df_sorted.loc[idx_teste, time_col]

    treino = df[df[time_col] < cutoff_tuning].copy()
    tuning = df[(df[time_col] >= cutoff_tuning) & (df[time_col] < cutoff_teste)].copy()
    teste = df[df[time_col] >= cutoff_teste].copy()

    return {
        "treino": treino,
        "tuning": tuning,
        "teste": teste,
        "cutoff_tuning": cutoff_tuning,
        "cutoff_teste": cutoff_teste,
        "frac_treino": frac_treino,
        "frac_tuning": frac_tuning,
        "frac_teste": 1 - frac_treino - frac_tuning,
    }


def aprender_ranking_sistemas(df_disparo_periodo, df_aparicoes_fonte_base,
                              metodo_atribuicao="full", metodo_ranking="empirical_bayes"):
    """Aprende ranking causal de sistemas a partir de um período de disparos."""
    df_join = join_disparo_sistema(df_disparo_periodo, df_aparicoes_fonte_base, causal=True)
    metricas = calcular_score_sistema(
        calcular_metricas_sistema(df_join, metodo_atribuicao=metodo_atribuicao),
        metodo_ranking=metodo_ranking,
    )
    return metricas, df_join


def anexar_score_sistema(df_aparicoes_fonte_base, metricas_sistema):
    """Mescla o score_sistema nas aparições por fonte."""
    out = df_aparicoes_fonte_base.merge(
        metricas_sistema[["id_sistema", "score_sistema"]],
        on="id_sistema", how="left",
    )
    fallback = metricas_sistema["score_sistema"].min() if len(metricas_sistema) else 0.0
    out["score_sistema"] = out["score_sistema"].fillna(fallback)
    return out


def montar_eventos(df_disparo_base, df_aparicoes_scored, df_meta, half_life):
    """
    Monta base de eventos com features por telefone para o modelo de priorização.

    Inclui: score de origem, decaimento temporal, score_qualidade, DDD,
    penalidade de proprietários, quantidade de sistemas e proporção causal.
    """
    df = (
        df_disparo_base.merge(
            df_aparicoes_scored[["telefone_numero", "id_sistema", "registro_data_atualizacao", "score_sistema"]],
            left_on="contato_telefone", right_on="telefone_numero", how="inner",
        )
        .merge(
            df_meta[["telefone_numero", "telefone_ddd", "n_proprietarios", "penalidade_proprietarios",
                      "n_sistemas_telefone", "score_qualidade", "is_ddd_21",
                      "cpfs_distintos_telefone", "score_exclusividade_cpf",
                      "log_cpfs_distintos_telefone"]],
            on="telefone_numero", how="left",
        )
    )
    df = adicionar_features_temporais(df, half_life, reference_col="envio_datahora")
    df = _selecionar_aparicoes_evento_fonte(df, causal_only=False)
    df["id_sistema_causal"] = df["id_sistema"].where(df["tem_data_causal"])
    df["score_sistema_causal"] = df["score_sistema"].where(df["tem_data_causal"], 0.0)
    df["decaimento_causal"] = df["decaimento_temporal"].where(df["tem_data_causal"], 0.0)
    df["score_aparicao_causal"] = df["score_sistema_causal"] * df["decaimento_causal"]

    eventos = df.groupby([
        "id_disparo", "cpf", "contato_telefone", "envio_datahora", "status_disparo",
        "telefone_ddd", "n_proprietarios", "penalidade_proprietarios",
        "n_sistemas_telefone", "score_qualidade", "is_ddd_21",
        "cpfs_distintos_telefone", "score_exclusividade_cpf", "log_cpfs_distintos_telefone",
    ], as_index=False, dropna=False).agg(
        max_score_origem_tempo=("score_aparicao_causal", "max"),
        melhor_score_sistema=("score_sistema_causal", "max"),
        melhor_decaimento=("decaimento_causal", "max"),
        melhor_dias_atualizacao=("dias_desde_atualizacao", "min"),
        media_dias_atualizacao=("dias_desde_atualizacao", "mean"),
        decaimento_medio=("decaimento_causal", "mean"),
        proporcao_aparicoes_causais=("tem_data_causal", "mean"),
        qtd_sistemas_candidatos=("id_sistema_causal", "nunique"),
    )

    causais = df[df["tem_data_causal"]].copy()
    if not causais.empty:
        fonte_recente = (
            causais.sort_values(["id_disparo", "registro_data_atualizacao", "score_sistema_causal"], na_position="first")
            .drop_duplicates("id_disparo", keep="last")
            [["id_disparo", "id_sistema", "score_sistema_causal", "dias_desde_atualizacao", "decaimento_causal"]]
            .rename(columns={
                "id_sistema": "id_sistema_fonte_mais_recente",
                "score_sistema_causal": "score_fonte_mais_recente",
                "dias_desde_atualizacao": "dias_fonte_mais_recente",
                "decaimento_causal": "decaimento_fonte_mais_recente",
            })
        )
        melhor_fonte = (
            causais.sort_values(["id_disparo", "score_aparicao_causal", "registro_data_atualizacao"], na_position="first")
            .drop_duplicates("id_disparo", keep="last")
            [["id_disparo", "dias_desde_atualizacao", "decaimento_causal"]]
            .rename(columns={
                "dias_desde_atualizacao": "dias_melhor_fonte",
                "decaimento_causal": "decaimento_melhor_fonte",
            })
        )
        eventos = eventos.merge(fonte_recente, on="id_disparo", how="left")
        eventos = eventos.merge(melhor_fonte, on="id_disparo", how="left")
    else:
        eventos["id_sistema_fonte_mais_recente"] = np.nan
        eventos["score_fonte_mais_recente"] = 0.0
        eventos["dias_fonte_mais_recente"] = 9999
        eventos["decaimento_fonte_mais_recente"] = 0.0
        eventos["dias_melhor_fonte"] = 9999
        eventos["decaimento_melhor_fonte"] = 0.0

    eventos["telefone_ddd"] = eventos["telefone_ddd"].fillna(-1)
    eventos["is_ddd_21"] = eventos["is_ddd_21"].fillna(0).astype(int)
    eventos["n_proprietarios"] = eventos["n_proprietarios"].fillna(1).clip(lower=1)
    eventos["penalidade_proprietarios"] = eventos["penalidade_proprietarios"].fillna(1.0)
    eventos["n_sistemas_telefone"] = eventos["qtd_sistemas_candidatos"].fillna(0).astype(int)
    eventos["log_n_sistemas_telefone"] = np.log1p(eventos["n_sistemas_telefone"])
    eventos["score_qualidade"] = eventos["score_qualidade"].fillna(QUALIDADE_DEFAULT)
    eventos["cpfs_distintos_telefone"] = eventos["cpfs_distintos_telefone"].fillna(1).clip(lower=1)
    eventos["score_exclusividade_cpf"] = eventos["score_exclusividade_cpf"].fillna(1.0)
    eventos["log_cpfs_distintos_telefone"] = eventos["log_cpfs_distintos_telefone"].fillna(
        np.log1p(eventos["cpfs_distintos_telefone"])
    )
    eventos["score_fonte_mais_recente"] = eventos["score_fonte_mais_recente"].fillna(0.0)
    eventos["dias_fonte_mais_recente"] = eventos["dias_fonte_mais_recente"].fillna(9999)
    eventos["decaimento_fonte_mais_recente"] = eventos["decaimento_fonte_mais_recente"].fillna(0.0)
    eventos["dias_melhor_fonte"] = eventos["dias_melhor_fonte"].fillna(9999)
    eventos["decaimento_melhor_fonte"] = eventos["decaimento_melhor_fonte"].fillna(0.0)
    eventos["media_dias_atualizacao"] = eventos["media_dias_atualizacao"].fillna(9999)
    eventos["decaimento_medio"] = eventos["decaimento_medio"].fillna(0.0)
    eventos["y_entrega"] = eventos["status_disparo"].isin(STATUS_ENTREGA).astype(int)
    eventos["y_read"] = (eventos["status_disparo"] == "read").astype(int)
    return eventos


def preparar_matrizes_modelo(df_eventos, cutoff_time_ref):
    """
    Split temporal treino/validação e cálculo de prior suavizado por DDD.

    Retorna (df_train, df_val, feature_cols, prior_ddd, baseline_ddd).

    NOTA: o cutoff de validação é compartilhado com a seleção de half-life,
    o que pode inflacionar as estimativas do holdout. Isso é uma limitação
    conhecida e documentada; a decisão final fica para o A/B test.
    """
    df_train = df_eventos[df_eventos["envio_datahora"] < cutoff_time_ref].copy()
    df_val = df_eventos[df_eventos["envio_datahora"] >= cutoff_time_ref].copy()

    prior_ddd, baseline_ddd = calcular_prior_suavizado(df_train, "telefone_ddd", "y_entrega")
    df_train = df_train.merge(prior_ddd, on="telefone_ddd", how="left").rename(columns={"score_prior": "score_ddd"})
    df_val = df_val.merge(prior_ddd, on="telefone_ddd", how="left").rename(columns={"score_prior": "score_ddd"})
    df_train["score_ddd"] = df_train["score_ddd"].fillna(baseline_ddd)
    df_val["score_ddd"] = df_val["score_ddd"].fillna(baseline_ddd)

    return df_train, df_val, FEATURE_COLS, prior_ddd, baseline_ddd


def score_phones_at_reference(df_aparicoes_scored, df_meta, reference_time, half_life,
                               prior_ddd, baseline_ddd, scaler, modelo, feature_cols):
    """
    Gera score operacional por telefone em um instante de referência.

    Produz score_modelo (logístico) e score_heurístico (baseline arbitrário).
    """
    df = df_aparicoes_scored.copy()
    df["reference_time"] = reference_time
    df = adicionar_features_temporais(df, half_life, reference_col="reference_time")
    keys = ["telefone_numero", "id_sistema"]
    causal = (
        df[df["tem_data_causal"]]
        .sort_values(keys + ["registro_data_atualizacao"], na_position="first")
        .drop_duplicates(keys, keep="last")
    )
    causal_keys = causal[keys].drop_duplicates()
    sem_causal = df.merge(causal_keys.assign(_tem_aparicao_causal=1), on=keys, how="left")
    sem_causal = sem_causal[sem_causal["_tem_aparicao_causal"].isna()].drop(columns=["_tem_aparicao_causal"])
    nao_causal = (
        sem_causal.sort_values(keys + ["registro_data_atualizacao"], na_position="first")
        .drop_duplicates(keys, keep="last")
    )
    df = pd.concat([causal, nao_causal], ignore_index=True)
    df["id_sistema_causal"] = df["id_sistema"].where(df["tem_data_causal"])
    df["score_sistema_causal"] = df["score_sistema"].where(df["tem_data_causal"], 0.0)
    df["decaimento_causal"] = df["decaimento_temporal"].where(df["tem_data_causal"], 0.0)
    df["score_aparicao_causal"] = df["score_sistema_causal"] * df["decaimento_causal"]

    telefones = df.groupby("telefone_numero", as_index=False).agg(
        max_score_origem_tempo=("score_aparicao_causal", "max"),
        melhor_score_sistema=("score_sistema_causal", "max"),
        melhor_decaimento=("decaimento_causal", "max"),
        melhor_dias_atualizacao=("dias_desde_atualizacao", "min"),
        media_dias_atualizacao=("dias_desde_atualizacao", "mean"),
        decaimento_medio=("decaimento_causal", "mean"),
        proporcao_aparicoes_causais=("tem_data_causal", "mean"),
        qtd_sistemas_candidatos=("id_sistema_causal", "nunique"),
    )
    causais = df[df["tem_data_causal"]].copy()
    if not causais.empty:
        fonte_recente = (
            causais.sort_values(["telefone_numero", "registro_data_atualizacao", "score_sistema_causal"], na_position="first")
            .drop_duplicates("telefone_numero", keep="last")
            [["telefone_numero", "id_sistema", "score_sistema_causal", "dias_desde_atualizacao", "decaimento_causal"]]
            .rename(columns={
                "id_sistema": "id_sistema_fonte_mais_recente",
                "score_sistema_causal": "score_fonte_mais_recente",
                "dias_desde_atualizacao": "dias_fonte_mais_recente",
                "decaimento_causal": "decaimento_fonte_mais_recente",
            })
        )
        melhor_fonte = (
            causais.sort_values(["telefone_numero", "score_aparicao_causal", "registro_data_atualizacao"], na_position="first")
            .drop_duplicates("telefone_numero", keep="last")
            [["telefone_numero", "dias_desde_atualizacao", "decaimento_causal"]]
            .rename(columns={
                "dias_desde_atualizacao": "dias_melhor_fonte",
                "decaimento_causal": "decaimento_melhor_fonte",
            })
        )
        telefones = telefones.merge(fonte_recente, on="telefone_numero", how="left")
        telefones = telefones.merge(melhor_fonte, on="telefone_numero", how="left")
    else:
        telefones["id_sistema_fonte_mais_recente"] = np.nan
        telefones["score_fonte_mais_recente"] = 0.0
        telefones["dias_fonte_mais_recente"] = 9999
        telefones["decaimento_fonte_mais_recente"] = 0.0
        telefones["dias_melhor_fonte"] = 9999
        telefones["decaimento_melhor_fonte"] = 0.0
    telefones = telefones.merge(
        df_meta[["telefone_numero", "telefone_ddd", "n_proprietarios",
                   "penalidade_proprietarios", "n_sistemas_telefone", "score_qualidade",
                   "is_ddd_21", "cpfs_distintos_telefone", "score_exclusividade_cpf",
                   "log_cpfs_distintos_telefone"]],
        on="telefone_numero", how="left",
    )
    telefones["telefone_ddd"] = telefones["telefone_ddd"].fillna(-1)
    telefones["is_ddd_21"] = telefones["is_ddd_21"].fillna(0).astype(int)
    telefones["n_proprietarios"] = telefones["n_proprietarios"].fillna(1).clip(lower=1)
    telefones["penalidade_proprietarios"] = telefones["penalidade_proprietarios"].fillna(1.0)
    telefones["n_sistemas_telefone"] = telefones["qtd_sistemas_candidatos"].fillna(0).astype(int)
    telefones["log_n_sistemas_telefone"] = np.log1p(telefones["n_sistemas_telefone"])
    telefones["score_qualidade"] = telefones["score_qualidade"].fillna(QUALIDADE_DEFAULT)
    telefones["cpfs_distintos_telefone"] = telefones["cpfs_distintos_telefone"].fillna(1).clip(lower=1)
    telefones["score_exclusividade_cpf"] = telefones["score_exclusividade_cpf"].fillna(1.0)
    telefones["log_cpfs_distintos_telefone"] = telefones["log_cpfs_distintos_telefone"].fillna(
        np.log1p(telefones["cpfs_distintos_telefone"])
    )
    telefones["score_fonte_mais_recente"] = telefones["score_fonte_mais_recente"].fillna(0.0)
    telefones["dias_fonte_mais_recente"] = telefones["dias_fonte_mais_recente"].fillna(9999)
    telefones["decaimento_fonte_mais_recente"] = telefones["decaimento_fonte_mais_recente"].fillna(0.0)
    telefones["dias_melhor_fonte"] = telefones["dias_melhor_fonte"].fillna(9999)
    telefones["decaimento_melhor_fonte"] = telefones["decaimento_melhor_fonte"].fillna(0.0)
    telefones["media_dias_atualizacao"] = telefones["media_dias_atualizacao"].fillna(9999)
    telefones["decaimento_medio"] = telefones["decaimento_medio"].fillna(0.0)
    telefones = telefones.merge(prior_ddd.rename(columns={"score_prior": "score_ddd"}), on="telefone_ddd", how="left")
    telefones["score_ddd"] = telefones["score_ddd"].fillna(baseline_ddd)

    X_score = telefones[feature_cols].fillna(0)
    telefones["score_modelo"] = modelo.predict_proba(scaler.transform(X_score))[:, 1]

    telefones["score_heuristico"] = sum(
        peso * normalizar_0_1(telefones[col])
        for col, peso in PESOS_HEURISTICA.items()
    )

    return telefones


def selecionar_top2(df_base, metodo, sort_cols, ascending):
    """Seleciona os 2 melhores telefones por CPF de acordo com critério de ordenação."""
    selecionados = (
        df_base.sort_values(["cpf"] + sort_cols, ascending=[True] + ascending)
        .groupby("cpf")
        .head(2)
        .copy()
    )
    selecionados["metodo"] = metodo
    selecionados["rank"] = selecionados.groupby("cpf").cumcount() + 1
    return selecionados


def gerar_selecoes(df_candidatos, incluir_random=True):
    """Gera seleções de top-2 por CPF usando múltiplos métodos."""
    selecoes = [
        selecionar_top2(df_candidatos, "modelo",
                        ["score_modelo", "score_heuristico", "melhor_dias_atualizacao", "telefone_numero"],
                        [False, False, True, True]),
        selecionar_top2(df_candidatos, "heuristica",
                        ["score_heuristico", "melhor_dias_atualizacao", "telefone_numero"],
                        [False, True, True]),
        selecionar_top2(df_candidatos, "mais_recente",
                        ["melhor_dias_atualizacao", "score_heuristico", "telefone_numero"],
                        [True, False, True]),
        selecionar_top2(df_candidatos, "fonte_mais_recente",
                        ["score_fonte_mais_recente", "dias_fonte_mais_recente", "score_heuristico", "telefone_numero"],
                        [False, True, False, True]),
        selecionar_top2(df_candidatos, "alfabetico",
                        ["telefone_numero"],
                        [True]),
    ]
    if incluir_random:
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            temp = df_candidatos.copy()
            temp["ordem_randomica"] = rng.random(len(temp))
            selecoes.append(selecionar_top2(temp, f"random_{seed:02d}", ["ordem_randomica"], [True]))
    return pd.concat(selecoes, ignore_index=True)


def prob_ao_menos_um_sucesso(series):
    """Probabilidade de ao menos um sucesso em eventos independentes."""
    valores = series.dropna().clip(0, 1)
    if valores.empty:
        return np.nan
    return 1 - np.prod(1 - valores)


def soma_com_evidencia(series):
    """Soma de valores não-nulos."""
    valores = series.dropna()
    if valores.empty:
        return np.nan
    return valores.sum()


def avaliar_selecao(df_selecoes, df_holdout):
    """Avalia seleções de telefones contra holdout histórico."""
    aval = df_selecoes.merge(df_holdout, on="telefone_numero", how="left")
    resumo = aval.groupby("metodo").agg(
        cpfs=("cpf", "nunique"),
        telefones_escolhidos=("telefone_numero", "count"),
        cobertura_holdout=("total_validacao", lambda s: s.notna().mean()),
        taxa_entrega_media_top2=("taxa_entrega_validacao", "mean"),
        taxa_read_media_top2=("taxa_read_validacao", "mean"),
    ).reset_index()
    top1 = aval[aval["rank"] == 1].groupby("metodo")[["taxa_entrega_validacao", "taxa_read_validacao"]].mean().reset_index().rename(columns={
        "taxa_entrega_validacao": "taxa_entrega_top1",
        "taxa_read_validacao": "taxa_read_top1",
    })

    cpf_top2 = aval.groupby(["metodo", "cpf"]).agg(
        telefones_escolhidos=("telefone_numero", "count"),
        telefones_com_evidencia=("total_validacao", lambda s: s.notna().sum()),
        entrega_esperada_top2=("taxa_entrega_validacao", soma_com_evidencia),
        leitura_esperada_top2=("taxa_read_validacao", soma_com_evidencia),
        prob_ao_menos_uma_entrega_proxy=("taxa_entrega_validacao", prob_ao_menos_um_sucesso),
        prob_ao_menos_um_read_proxy=("taxa_read_validacao", prob_ao_menos_um_sucesso),
    ).reset_index()
    cpf_top2["tem_evidencia_holdout"] = cpf_top2["telefones_com_evidencia"] > 0

    metricas_top2_cpf = cpf_top2.groupby("metodo").agg(
        cpfs=("cpf", "nunique"),
        cobertura_cpf_holdout=("tem_evidencia_holdout", "mean"),
        telefones_com_evidencia_media=("telefones_com_evidencia", "mean"),
        entrega_esperada_top2=("entrega_esperada_top2", "mean"),
        leitura_esperada_top2=("leitura_esperada_top2", "mean"),
        prob_ao_menos_uma_entrega_proxy=("prob_ao_menos_uma_entrega_proxy", "mean"),
        prob_ao_menos_um_read_proxy=("prob_ao_menos_um_read_proxy", "mean"),
    ).reset_index()

    return resumo.merge(top1, on="metodo", how="left"), aval, metricas_top2_cpf, cpf_top2


def bootstrap_comparacao_metodos(metricas_cpf, metodo_referencia, comparadores,
                                  metric_col="prob_ao_menos_uma_entrega_proxy",
                                  n_boot=200, seed=SEED):
    """Compara o método referência contra cada comparador via bootstrap."""
    pivot = metricas_cpf.pivot_table(index="cpf", columns="metodo", values=metric_col)
    random_cols = [col for col in pivot.columns if str(col).startswith("random_")]
    if random_cols:
        pivot["random_media_20_seeds"] = pivot[random_cols].mean(axis=1)

    rng = np.random.default_rng(seed)
    rows = []
    for comparador in comparadores:
        if metodo_referencia not in pivot.columns or comparador not in pivot.columns:
            continue
        pares = pivot[[metodo_referencia, comparador]].dropna()
        if pares.empty:
            continue
        diff = pares[metodo_referencia].to_numpy() - pares[comparador].to_numpy()
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(diff), len(diff))
            boot.append(diff[idx].mean())
        rows.append({
            "metrica": metric_col,
            "metodo_referencia": metodo_referencia,
            "comparador": comparador,
            "n_cpfs": len(diff),
            "diff_media": diff.mean(),
            "ci95_low": np.percentile(boot, 2.5),
            "ci95_high": np.percentile(boot, 97.5),
        })
    return pd.DataFrame(rows)


def calcular_metricas_por_categoria(df_disparo_sistema, df_disparo_base=None):
    """
    Calcula métricas de entrega por (sistema, categoria_hsm).

    Permite verificar se o desempenho de um sistema varia
    significativamente entre categorias de campanha.
    """
    if "categoria_hsm" in df_disparo_sistema.columns:
        df = df_disparo_sistema.copy()
    elif df_disparo_base is not None and "categoria_hsm" in df_disparo_base.columns:
        df = df_disparo_sistema.merge(
            df_disparo_base[["id_disparo", "categoria_hsm"]].drop_duplicates(),
            on="id_disparo", how="left",
        )
    else:
        df = df_disparo_sistema

    if "categoria_hsm" not in df.columns:
        print("Coluna categoria_hsm não disponível para quebra.")
        return None

    rows = []
    for (sistema, categoria), grupo in df.groupby(["id_sistema", "categoria_hsm"], dropna=False):
        total = grupo["id_disparo"].nunique()
        entregas = grupo.loc[grupo["status_disparo"].isin(STATUS_ENTREGA), "id_disparo"].nunique()
        rows.append({
            "id_sistema": sistema,
            "categoria_hsm": categoria,
            "total": total,
            "entregas": entregas,
            "taxa_entrega": entregas / total if total > 0 else np.nan,
        })
    return pd.DataFrame(rows)
