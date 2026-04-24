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
    wilson_lower_bound,
    normalizar_0_1,
    calcular_score_sistema,
    calcular_metricas_sistema,
    preparar_metadados_telefone,
    adicionar_features_temporais,
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

    def test_vazio(self):
        df = pd.DataFrame(columns=["id_sistema", "id_disparo", "status_disparo"])
        metricas = calcular_metricas_sistema(df)
        assert metricas.empty


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
        meta = preparar_metadados_telefone(df_telefone, df_aparicoes)
        assert "score_qualidade" in meta.columns
        assert meta.loc[meta["telefone_numero"] == 1, "score_qualidade"].values[0] == 1.0
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