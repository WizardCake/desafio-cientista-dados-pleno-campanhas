"""
Funções utilitárias compartilhadas entre os notebooks do desafio.

A métrica operacional primária desta solução é entrega:
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
    "penalidade_proprietarios",
    "score_ddd",
    "score_qualidade",
    "log_n_sistemas_telefone",
    "proporcao_aparicoes_causais",
]

PESOS_HEURISTICA = {
    "max_score_origem_tempo": 0.45,
    "score_qualidade": 0.10,
    "score_ddd": 0.20,
    "penalidade_proprietarios": 0.15,
    "log_n_sistemas_telefone": 0.05,
    "proporcao_aparicoes_causais": 0.05,
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
    Uma linha por (telefone_numero, id_sistema).

    Mantém a data de atualização mais recente para evitar que um telefone
    com muitas aparições no mesmo sistema infle o ranking da fonte.
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


def preparar_metadados_telefone(df_telefone, df_aparicoes_fonte):
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
    return meta


# ============================================================
# JOIN E MÉTRICAS
# ============================================================

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

    if causal:
        envio = pd.to_datetime(df["envio_datahora"])
        atualizacao = pd.to_datetime(df["registro_data_atualizacao"])
        df = df[atualizacao.notna() & (atualizacao <= envio)].copy()

    n_disparos_depois = df["id_disparo"].nunique()
    n_linhas = len(df)
    multiplicidade = n_linhas / n_disparos_depois if n_disparos_depois else 0

    print(f"Disparos com match: {n_disparos_depois:,} / {n_disparos_antes:,}")
    print(f"Total de linhas após join: {n_linhas:,}")
    print(f"Multiplicidade média: {multiplicidade:.2f}x")
    return df


def _contar_status_unico(grupo, status):
    return grupo.loc[grupo["status_disparo"].eq(status), "id_disparo"].nunique()


def calcular_metricas_sistema(df_disparo_sistema):
    """Calcula métricas por sistema com denominador em disparos únicos."""
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
    """Limite inferior do intervalo de confiança de Wilson."""
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
    """Normalização min-max robusta para séries constantes."""
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

def aprender_ranking_sistemas(df_disparo_periodo, df_aparicoes_fonte_base):
    """Aprende ranking causal de sistemas a partir de um período de disparos."""
    df_join = join_disparo_sistema(df_disparo_periodo, df_aparicoes_fonte_base, causal=True)
    metricas = calcular_score_sistema(calcular_metricas_sistema(df_join))
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
                      "n_sistemas_telefone", "score_qualidade"]],
            on="telefone_numero", how="left",
        )
    )
    df = adicionar_features_temporais(df, half_life, reference_col="envio_datahora")
    df["id_sistema_causal"] = df["id_sistema"].where(df["tem_data_causal"])
    df["score_sistema_causal"] = df["score_sistema"].where(df["tem_data_causal"], 0.0)
    df["decaimento_causal"] = df["decaimento_temporal"].where(df["tem_data_causal"], 0.0)
    df["score_aparicao_causal"] = df["score_sistema_causal"] * df["decaimento_causal"]

    eventos = df.groupby([
        "id_disparo", "cpf", "contato_telefone", "envio_datahora", "status_disparo",
        "telefone_ddd", "n_proprietarios", "penalidade_proprietarios",
        "n_sistemas_telefone", "score_qualidade",
    ], as_index=False, dropna=False).agg(
        max_score_origem_tempo=("score_aparicao_causal", "max"),
        melhor_score_sistema=("score_sistema_causal", "max"),
        melhor_decaimento=("decaimento_causal", "max"),
        melhor_dias_atualizacao=("dias_desde_atualizacao", "min"),
        proporcao_aparicoes_causais=("tem_data_causal", "mean"),
        qtd_sistemas_candidatos=("id_sistema_causal", "nunique"),
    )

    eventos["telefone_ddd"] = eventos["telefone_ddd"].fillna(-1)
    eventos["n_proprietarios"] = eventos["n_proprietarios"].fillna(1).clip(lower=1)
    eventos["penalidade_proprietarios"] = eventos["penalidade_proprietarios"].fillna(1.0)
    eventos["n_sistemas_telefone"] = eventos["qtd_sistemas_candidatos"].fillna(0).astype(int)
    eventos["log_n_sistemas_telefone"] = np.log1p(eventos["n_sistemas_telefone"])
    eventos["score_qualidade"] = eventos["score_qualidade"].fillna(QUALIDADE_DEFAULT)
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
    df["id_sistema_causal"] = df["id_sistema"].where(df["tem_data_causal"])
    df["score_sistema_causal"] = df["score_sistema"].where(df["tem_data_causal"], 0.0)
    df["decaimento_causal"] = df["decaimento_temporal"].where(df["tem_data_causal"], 0.0)
    df["score_aparicao_causal"] = df["score_sistema_causal"] * df["decaimento_causal"]

    telefones = df.groupby("telefone_numero", as_index=False).agg(
        max_score_origem_tempo=("score_aparicao_causal", "max"),
        melhor_score_sistema=("score_sistema_causal", "max"),
        melhor_decaimento=("decaimento_causal", "max"),
        melhor_dias_atualizacao=("dias_desde_atualizacao", "min"),
        proporcao_aparicoes_causais=("tem_data_causal", "mean"),
        qtd_sistemas_candidatos=("id_sistema_causal", "nunique"),
    )
    telefones = telefones.merge(
        df_meta[["telefone_numero", "telefone_ddd", "n_proprietarios",
                   "penalidade_proprietarios", "n_sistemas_telefone", "score_qualidade"]],
        on="telefone_numero", how="left",
    )
    telefones["telefone_ddd"] = telefones["telefone_ddd"].fillna(-1)
    telefones["n_proprietarios"] = telefones["n_proprietarios"].fillna(1).clip(lower=1)
    telefones["penalidade_proprietarios"] = telefones["penalidade_proprietarios"].fillna(1.0)
    telefones["n_sistemas_telefone"] = telefones["qtd_sistemas_candidatos"].fillna(0).astype(int)
    telefones["log_n_sistemas_telefone"] = np.log1p(telefones["n_sistemas_telefone"])
    telefones["score_qualidade"] = telefones["score_qualidade"].fillna(QUALIDADE_DEFAULT)
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
    if df_disparo_base is not None and "categoria_hsm" in df_disparo_base.columns:
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