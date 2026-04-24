import numpy as np
import pandas as pd
import pytest

from src.utils import (
    QUALIDADE_DEFAULT,
    QUALIDADE_MAP,
    FEATURE_COLS,
    PESOS_HEURISTICA,
    PESOS_HEURISTICA_NOTA,
    STATUS_ENTREGA,
    STATUS_INVALIDOS,
    TIPOS_FIXOS,
    empirical_bayes_lower_bound,
    wilson_lower_bound,
    normalizar_0_1,
    calcular_score_sistema,
    calcular_metricas_sistema,
    preparar_metadados_telefone,
    adicionar_features_temporais,
    join_disparo_sistema,
    criar_splits_temporais,
    selecionar_top2,
    prob_ao_menos_um_sucesso,
    soma_com_evidencia,
)


class TestConstantes:
    def test_status_entrega(self):
        assert STATUS_ENTREGA == {"delivered", "read"}

    def test_status_invalidos(self):
        assert STATUS_INVALIDOS == {"processing"}

    def test_tipos_fixos(self):
        assert TIPOS_FIXOS == {"fixo", "fixa"}

    def test_qualidade_map_valores(self):
        assert QUALIDADE_MAP["ALTA"] == 1.0
        assert QUALIDADE_MAP["MEDIA"] == 0.5
        assert QUALIDADE_MAP["BAIXA"] == 0.0

    def test_qualidade_default(self):
        assert 0.0 < QUALIDADE_DEFAULT < 0.5

    def test_feature_cols_contem_qualidade(self):
        assert "score_qualidade" in FEATURE_COLS
        assert "score_fonte_mais_recente" in FEATURE_COLS
        assert "score_exclusividade_cpf" in FEATURE_COLS
        assert "is_ddd_21" in FEATURE_COLS

    def test_pesos_heuristica_soma_um(self):
        total = sum(PESOS_HEURISTICA.values())
        assert abs(total - 1.0) < 1e-9

    def test_pesos_heuristica_nota_nao_vazia(self):
        assert len(PESOS_HEURISTICA_NOTA) > 0

    def test_feature_cols_subset_pesos(self):
        for col in PESOS_HEURISTICA:
            assert col in FEATURE_COLS


class TestWilsonLowerBound:
    def test_total_zero(self):
        assert wilson_lower_bound(0, 0) == 0.0

    def test_perfeita(self):
        result = wilson_lower_bound(100, 100, alpha=0.05)
        assert 0.9 < result < 1.0

    def test_pessima(self):
        result = wilson_lower_bound(0, 100, alpha=0.05)
        assert result < 0.05

    def test_volume_baixo_penaliza(self):
        lb_low = wilson_lower_bound(1, 2, alpha=0.05)
        lb_high = wilson_lower_bound(50, 100, alpha=0.05)
        assert lb_low < lb_high


class TestEmpiricalBayesLowerBound:
    def test_total_zero_global(self):
        assert empirical_bayes_lower_bound(0, 0, 0, 0) == 0.0

    def test_shrinkage_fonte_pequena(self):
        pequena = empirical_bayes_lower_bound(1, 1, 90, 100, prior_strength=20)
        grande = empirical_bayes_lower_bound(90, 100, 90, 100, prior_strength=20)
        assert pequena < 1.0
        assert grande > pequena


class TestNormalizar01:
    def test_constante_retorna_uns(self):
        s = pd.Series([5, 5, 5])
        result = normalizar_0_1(s)
        assert (result == 1.0).all()

    def test_min_max(self):
        s = pd.Series([10, 20, 30])
        result = normalizar_0_1(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[2] == 1.0
        assert abs(result.iloc[1] - 0.5) < 1e-9

    def test_com_nan(self):
        s = pd.Series([1, np.nan, 3])
        result = normalizar_0_1(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[2] == 1.0


class TestCalcularMetricasSistema:
    def test_simples(self):
        df = pd.DataFrame({
            "id_sistema": ["A", "A", "B", "B"],
            "id_disparo": [1, 2, 3, 4],
            "status_disparo": ["delivered", "failed", "read", "sent"],
        })
        metricas = calcular_metricas_sistema(df)
        assert len(metricas) == 2
        assert "taxa_entrega" in metricas.columns

    def test_fracionario_divide_credito_entre_fontes(self):
        df = pd.DataFrame({
            "id_sistema": ["A", "B", "A"],
            "id_disparo": [1, 1, 2],
            "status_disparo": ["read", "read", "failed"],
            "registro_data_atualizacao": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
        })
        metricas = calcular_metricas_sistema(df, metodo_atribuicao="fracionario")
        total_a = metricas.loc[metricas["id_sistema"] == "A", "total_disparos"].iloc[0]
        sucesso_a = metricas.loc[metricas["id_sistema"] == "A", "sucessos_entrega"].iloc[0]
        assert abs(total_a - 1.5) < 1e-9
        assert abs(sucesso_a - 0.5) < 1e-9

    def test_fonte_mais_recente_atribui_um_sistema_por_disparo(self):
        df = pd.DataFrame({
            "id_sistema": ["A", "B"],
            "id_disparo": [1, 1],
            "status_disparo": ["read", "read"],
            "registro_data_atualizacao": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-02-01"),
            ],
        })
        metricas = calcular_metricas_sistema(df, metodo_atribuicao="fonte_mais_recente")
        assert metricas["total_disparos"].sum() == 1.0
        assert metricas["id_sistema"].tolist() == ["B"]

    def test_vazio(self):
        df = pd.DataFrame(columns=["id_sistema", "id_disparo", "status_disparo"])
        metricas = calcular_metricas_sistema(df)
        assert metricas.empty


class TestCalcularScoreSistema:
    def test_score_empirical_bayes_padrao(self):
        metricas = pd.DataFrame({
            "id_sistema": ["A", "B"],
            "total_disparos": [100, 10],
            "read": [80, 9],
            "delivered": [10, 0],
            "failed": [10, 1],
            "sent": [0, 0],
            "sucessos_entrega": [90, 9],
            "taxa_entrega": [0.90, 0.90],
            "taxa_leitura": [0.80, 0.90],
            "taxa_falha": [0.10, 0.10],
        })
        result = calcular_score_sistema(metricas)
        assert "eb_lower_entrega" in result.columns
        assert "posterior_mean_entrega" in result.columns
        assert "wilson_lower_entrega" in result.columns
        assert result["metodo_ranking"].eq("empirical_bayes").all()


class TestPrepararMetadadosTelefone:
    def test_score_qualidade_adicionado(self):
        df_telefone = pd.DataFrame({
            "telefone_numero": [1, 2, 3, 4],
            "telefone_ddd": [21, 11, 31, 0],
            "telefone_proprietarios_quantidade": [1, 2, 1, np.nan],
            "telefone_sistemas_quantidade": [2, 1, 3, 1],
            "telefone_qualidade": ["ALTA", "MEDIA", "BAIXA", np.nan],
        })
        df_aparicoes = pd.DataFrame({
            "telefone_numero": [1, 2, 3],
            "id_sistema": ["S1", "S1", "S2"],
            "registro_data_atualizacao": [pd.Timestamp("2024-01-01")] * 3,
            "cpfs_sistema_distintos": [1, 1, 1],
            "qtd_aparicoes_brutas": [1, 1, 1],
        })
        df_aparicoes_brutas = pd.DataFrame({
            "telefone_numero": [1, 1, 2, 3],
            "cpf_sistema": ["cpf1", "cpf2", "cpf3", "cpf4"],
        })
        meta = preparar_metadados_telefone(df_telefone, df_aparicoes, df_aparicoes_brutas)
        assert "score_qualidade" in meta.columns
        assert "score_exclusividade_cpf" in meta.columns
        assert "is_ddd_21" in meta.columns
        assert meta.loc[meta["telefone_numero"] == 1, "score_qualidade"].values[0] == 1.0
        assert meta.loc[meta["telefone_numero"] == 1, "score_exclusividade_cpf"].values[0] == 0.5
        assert meta.loc[meta["telefone_numero"] == 1, "is_ddd_21"].values[0] == 1
        assert meta.loc[meta["telefone_numero"] == 2, "score_qualidade"].values[0] == 0.5
        assert meta.loc[meta["telefone_numero"] == 3, "score_qualidade"].values[0] == 0.0
        assert meta.loc[meta["telefone_numero"] == 4, "score_qualidade"].values[0] == QUALIDADE_DEFAULT


class TestAdicionarFeaturesTemporais:
    def test_causal(self):
        df = pd.DataFrame({
            "envio_datahora": [pd.Timestamp("2024-06-01")] * 3,
            "registro_data_atualizacao": [
                pd.Timestamp("2024-05-01"),
                pd.Timestamp("2024-07-01"),
                pd.NaT,
            ],
        })
        result = adicionar_features_temporais(df, half_life=90)
        assert result.loc[0, "tem_data_causal"] == True
        assert result.loc[1, "tem_data_causal"] == False
        assert result.loc[2, "tem_data_causal"] == False
        assert result.loc[0, "decaimento_temporal"] > 0
        assert result.loc[1, "dias_desde_atualizacao"] == 9999


class TestJoinDisparoSistema:
    def test_causal_usa_ultima_aparicao_antes_do_envio(self):
        df_disparo = pd.DataFrame({
            "id_disparo": [1],
            "contato_telefone": [10],
            "envio_datahora": [pd.Timestamp("2024-06-01")],
            "status_disparo": ["read"],
        })
        df_aparicoes = pd.DataFrame({
            "telefone_numero": [10, 10],
            "id_sistema": ["A", "A"],
            "registro_data_atualizacao": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-07-01"),
            ],
        })
        result = join_disparo_sistema(df_disparo, df_aparicoes, causal=True)
        assert len(result) == 1
        assert result["registro_data_atualizacao"].iloc[0] == pd.Timestamp("2024-01-01")


class TestCriarSplitsTemporais:
    def test_split_sem_vazamento_temporal(self):
        df = pd.DataFrame({
            "envio_datahora": pd.date_range("2024-01-01", periods=10, freq="D"),
            "id_disparo": range(10),
        })
        splits = criar_splits_temporais(df, frac_treino=0.6, frac_tuning=0.2)
        assert len(splits["treino"]) > 0
        assert len(splits["tuning"]) > 0
        assert len(splits["teste"]) > 0
        assert splits["treino"]["envio_datahora"].max() < splits["cutoff_tuning"]
        assert splits["tuning"]["envio_datahora"].min() >= splits["cutoff_tuning"]
        assert splits["teste"]["envio_datahora"].min() >= splits["cutoff_teste"]


class TestSelecionarTop2:
    def test_selecao_simples(self):
        df = pd.DataFrame({
            "cpf": [1, 1, 1, 2, 2],
            "telefone_numero": [10, 20, 30, 40, 50],
            "score_modelo": [0.9, 0.5, 0.3, 0.8, 0.7],
        })
        result = selecionar_top2(df, "teste", ["score_modelo"], [False])
        assert len(result) == 4
        assert result[result["cpf"] == 1]["telefone_numero"].tolist() == [10, 20]
        assert result[result["cpf"] == 2]["telefone_numero"].tolist() == [40, 50]


class TestProbAoMenosUmSucesso:
    def test_dois_independentes(self):
        s = pd.Series([0.5, 0.5])
        result = prob_ao_menos_um_sucesso(s)
        assert abs(result - 0.75) < 1e-9

    def test_um_sucesso_certo(self):
        s = pd.Series([1.0, 0.0])
        result = prob_ao_menos_um_sucesso(s)
        assert abs(result - 1.0) < 1e-9

    def test_vazio(self):
        s = pd.Series([], dtype=float)
        result = prob_ao_menos_um_sucesso(s)
        assert np.isnan(result)


class TestSomaComEvidencia:
    def test_simples(self):
        s = pd.Series([1.0, 2.0, np.nan])
        assert soma_com_evidencia(s) == 3.0

    def test_tudo_nan(self):
        s = pd.Series([np.nan, np.nan])
        assert np.isnan(soma_com_evidencia(s))
