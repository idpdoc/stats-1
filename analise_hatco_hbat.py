#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análise de Outliers e Dados Ausentes — HATCO / HBAT_MISSING
Saída: relatorio_analise.html + Gráficos no Padrão Estatístico SPSS
"""

__author__ = "Marcos Cícero"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Marcos Cícero"

import datetime
from typing import Any, Dict, List
import warnings

from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configurações ─────────────────────────────────────────────────────────────

# Caminhos e Nomes de Arquivos
ARQUIVO_HATCO = "HATCO.csv"         # Base de dados para análise de outliers
ARQUIVO_HBAT = "HBAT_MISSING.csv"   # Base de dados com dados ausentes
ARQUIVO_HTML = "relatorio_analise.html"  # Relatório final formatado

# Limiares Estatísticos para Detecção de Outliers
LIMIAR_ZSCORE = 2.5       # Desvios-padrão (limiar univariado e bivariado)
LIMIAR_PVALOR_CHI2 = 0.001  # Limite Chi-Quadrado para Mahalanobis (multivariado)

# ── Utilitários ───────────────────────────────────────────────────────────────

def carregar_csv(caminho: str) -> pd.DataFrame:
    """Carrega um arquivo CSV."""
    return pd.read_csv(caminho, sep=",")

def colunas_numericas(df: pd.DataFrame) -> List[str]:
    """Retorna as colunas numéricas do DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

# ── Lógica Estatística ────────────────────────────────────────────────────────

def detectar_outliers_univariados(df: pd.DataFrame, colunas: List[str]) -> Dict[str, list]:
    """Detecta outliers univariados usando Z-score de forma vetorizada."""
    resultado = {}
    
    for coluna in colunas:
        z_scores = stats.zscore(df[coluna], nan_policy='omit')
        outliers_idx = df.index[abs(z_scores) > LIMIAR_ZSCORE]
        resultado[coluna] = outliers_idx.tolist()
        
    return resultado

def detectar_outliers_bivariados(df: pd.DataFrame, col_x: str, col_y: str) -> pd.DataFrame:
    """Detecta outliers bivariados através do Z-score dos resíduos da regressão."""
    dados = df[[col_x, col_y]].dropna()
    
    try:
        slope, intercept, *_ = stats.linregress(dados[col_x], dados[col_y])
        
        valores_previstos = intercept + slope * dados[col_x]
        residuos = dados[col_y] - valores_previstos
        z_residuos = stats.zscore(residuos)
        
        is_outlier = abs(z_residuos) > LIMIAR_ZSCORE
        outliers = dados[is_outlier].copy()
        outliers["residuo_padronizado"] = z_residuos[is_outlier].round(4)
        
        return outliers
    except ValueError:
        return pd.DataFrame()

def calcular_mahalanobis(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """Calcula a distância de Mahalanobis de forma vetorizada e identifica outliers."""
    dados = df[colunas].dropna()
    gl = len(colunas)
    
    if len(dados) <= gl:
        return _retornar_df_vazio(df.index, gl)
    
    dados_centralizados = dados - dados.mean()
    cov_mat = np.cov(dados.values.T)
    
    try:
        inv_cov_mat = np.linalg.pinv(cov_mat)
        d2 = np.sum(dados_centralizados.values @ inv_cov_mat * dados_centralizados.values, axis=1)
    except (np.linalg.LinAlgError, ValueError):
        return _retornar_df_vazio(df.index, gl)
    
    pvalor = 1 - stats.chi2.cdf(d2, df=gl)
    
    resultado_parcial = pd.DataFrame({
        "D2": d2,
        "gl": gl,
        "pvalor": pvalor,
        "outlier": pvalor < LIMIAR_PVALOR_CHI2
    }, index=dados.index)
    
    return resultado_parcial.reindex(df.index).fillna({"outlier": False})

def _retornar_df_vazio(index: pd.Index, gl: int) -> pd.DataFrame:
    """Função auxiliar para manter o código limpo ao retornar estrutura padrão."""
    return pd.DataFrame({
        "D2": np.nan, "gl": gl, "pvalor": np.nan, "outlier": False
    }, index=index)

def substituir_pela_media_hbat_missing(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """Imputa valores ausentes pela média da coluna."""
    df_imputado = df.copy()
    for coluna in colunas:
        if df_imputado[coluna].isnull().any():
            df_imputado[coluna] = df_imputado[coluna].fillna(df_imputado[coluna].mean())
    return df_imputado

# ── Geração de Gráficos ───────────────────────────────────────────────────────

def gerar_boxplot(df: pd.DataFrame, colunas_uni: List[str]):
    plt.figure(figsize=(5.5, 4))
    plt.boxplot([df[c].dropna() for c in colunas_uni], labels=colunas_uni,
                flierprops=dict(marker='o', markerfacecolor='#8B0000', markersize=6, linestyle='none'))
    plt.title('Diagrama de Caixa (Boxplot) — Identificação de Outliers', fontsize=11, fontweight='bold', color='#1B365D')
    plt.ylabel('Escala Amostral')
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("boxplot_spss.png", dpi=120)
    plt.close()

def gerar_dispersao(df: pd.DataFrame, col_x: str, col_y: str):
    plt.figure(figsize=(5.5, 4))
    plt.scatter(df[col_x], df[col_y], color='#1B365D', alpha=0.7, edgecolors='none', label='Casos')
    m, b = np.polyfit(df[col_x], df[col_y], 1)
    x_vals = np.array([df[col_x].min(), df[col_x].max()])
    plt.plot(x_vals, m*x_vals + b, color='#8B0000', linewidth=1.5, label='Ajuste Linear (SPSS)')
    plt.title(f'Gráfico de Dispersão ({col_x} × {col_y})', fontsize=11, fontweight='bold', color='#1B365D')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("dispersao_spss.png", dpi=120)
    plt.close()

def gerar_qq_mahalanobis(maha: pd.DataFrame, colunas_maha: List[str]):
    gl_maha = len(colunas_maha)
    d2_ordenado = np.sort(maha["D2"].dropna().values)
    n_casos = len(d2_ordenado)
    quantis_teoricos = stats.chi2.ppf((np.arange(n_casos) + 0.5) / n_casos, df=gl_maha)
    
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(quantis_teoricos, d2_ordenado, color='#1B365D', alpha=0.7, edgecolors='none')
    lim_graf = max(quantis_teoricos.max(), d2_ordenado.max())
    plt.plot([0, lim_graf], [0, lim_graf], color='#8B0000', linestyle='--', linewidth=1.5, label='Normalidade Esperada')
    plt.title('Gráfico de Probabilidade Q-Q Qui-Quadrado', fontsize=11, fontweight='bold', color='#1B365D')
    plt.xlabel('Quantis Teóricos de $\\chi^2$')
    plt.ylabel('Distâncias Amostrais $D^2$')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("qqplot_spss.png", dpi=120)
    plt.close()

def gerar_grafico_missing(percentual_ausentes: pd.Series):
    missing_pct_plot = percentual_ausentes[percentual_ausentes > 0].sort_values(ascending=False)
    if not missing_pct_plot.empty:
        plt.figure(figsize=(6, 3.8))
        plt.bar(missing_pct_plot.index, missing_pct_plot.values, color='#4A90E2')
        plt.title('Gráfico de Perda de Dados Omissos por Indicador', fontsize=11, fontweight='bold', color='#1B365D')
        plt.ylabel('Percentual Incompleto (%)')
        plt.ylim(0, max(missing_pct_plot.values) + 5)
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig("missing_spss.png", dpi=120)
        plt.close()

# ── Preparação de Dados para Template ───────────────────────────────────────

def _mapear_nomes_colunas(df: pd.DataFrame) -> Dict[str, str]:
    """Cria um dicionário para traduzir nomes case-insensitive para o nome real no DF."""
    return {col.lower(): col for col in df.columns}

def preparar_contexto_hatco(df: pd.DataFrame) -> Dict[str, Any]:
    """Executa o pipeline completo de detecção e remoção de outliers para a base HATCO."""
    mapa_cols = _mapear_nomes_colunas(df)

    variaveis_escalares_hatco = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x9', 'x10']
    colunas_escalares = [mapa_cols[c] for c in variaveis_escalares_hatco if c in mapa_cols]
    
    colunas_uni = [mapa_cols[c] for c in ['x1', 'x2', 'x5'] if c in mapa_cols]
    coluna_x, coluna_y = mapa_cols.get('x1'), mapa_cols.get('x2')
    colunas_maha = sorted(colunas_escalares)

    outliers_uni = detectar_outliers_univariados(df, colunas_uni)
    outliers_biv = detectar_outliers_bivariados(df, coluna_x, coluna_y)
    maha = calcular_mahalanobis(df, colunas_maha)
    maha_out = maha[maha["outlier"]]

    indices_flagados = set().union(
        outliers_biv.index,
        maha_out.index,
        *[idx for idx in outliers_uni.values()]
    )

    df_sem_outliers = df.drop(index=indices_flagados, errors="ignore")

    return {
        "colunas_uni": colunas_uni,
        "coluna_x": coluna_x,
        "coluna_y": coluna_y,
        "colunas_maha": colunas_maha,
        "colunas_escalares": colunas_escalares,
        "outliers_uni": outliers_uni,
        "outliers_biv": outliers_biv,
        "maha": maha,
        "maha_out": maha_out,
        "indices_flagados": indices_flagados,
        "df_sem_outliers": df_sem_outliers,
        "n_original": len(df),
        "n_purificado": len(df_sem_outliers)
    }

def detecta_celulas_em_branco(df: pd.DataFrame) -> List[str]:
    """Filtra as variáveis escalares (v1 a v9) que possuem dados ausentes."""
    escalares = [f"v{i}" for i in range(1, 10)]
    return [col for col in escalares if col in df.columns and df[col].isnull().any()]

def preparar_contexto_hbat_missing(df: pd.DataFrame) -> Dict[str, Any]:
    """Analisa dados ausentes e aplica imputação pela média na base HBAT."""
    total_casos = len(df)
    
    contagem_ausentes = df.isnull().sum()
    percentual_ausentes = (df.isnull().mean() * 100).round(2)

    colunas_com_ausente = detecta_celulas_em_branco(df)
    df_imputado = substituir_pela_media_hbat_missing(df, colunas_com_ausente)

    tabela_missing = pd.DataFrame({
        "n_ausente": contagem_ausentes,
        "pct": percentual_ausentes,
        "n_valido": total_casos - contagem_ausentes
    }).rename_axis("variavel").reset_index()
    
    tabela_missing = tabela_missing.sort_values("pct", ascending=False)

    return {
        "total_casos": total_casos,
        "contagem_ausentes": contagem_ausentes,
        "percentual_ausentes": percentual_ausentes,
        "colunas_com_ausente": colunas_com_ausente,
        "df_imputado": df_imputado,
        "tabela_missing": tabela_missing
    }

# ── Renderização com Jinja2 ───────────────────────────────────────────────────

def gerar_relatorio():
    """Função principal que gera os gráficos e o relatório HTML."""
    try:
        df_hatco = carregar_csv(ARQUIVO_HATCO)
        df_hbat = carregar_csv(ARQUIVO_HBAT)
    except FileNotFoundError as e:
        print(f"❌ Erro ao carregar os arquivos: {e}")
        return

    ctx_hatco = preparar_contexto_hatco(df_hatco)
    ctx_hbat = preparar_contexto_hbat_missing(df_hbat)

    # Geração dos gráficos
    gerar_boxplot(df_hatco, ctx_hatco["colunas_uni"])
    gerar_dispersao(df_hatco, ctx_hatco["coluna_x"], ctx_hatco["coluna_y"])
    gerar_qq_mahalanobis(ctx_hatco["maha"], ctx_hatco["colunas_maha"])
    gerar_grafico_missing(ctx_hbat["percentual_ausentes"])

    # Consolidação do dicionário de contexto para o Jinja2
    contexto = {
        "titulo": "Diagnóstico Multivariado de Dados — Padrão SPSS",
        "limiar_zscore": LIMIAR_ZSCORE,
        "limiar_pvalor": LIMIAR_PVALOR_CHI2,
        "data_geracao": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "df_hatco": df_hatco,
        "df_hbat": df_hbat,
        **ctx_hatco,
        **ctx_hbat
    }

    # Inicialização do ambiente Jinja2 e gravação da saída
    try:
        env = Environment(loader=FileSystemLoader('.'), autoescape=True)
        template = env.get_template('template_relatorio.html')
        html = template.render(contexto)

        with open(ARQUIVO_HTML, "w", encoding="utf-8") as f:
            f.write(html)
            
        print(f"✅ Relatório gerado com sucesso: {ARQUIVO_HTML}")
        print("Gráficos gerados: boxplot_spss.png, dispersao_spss.png, qqplot_spss.png, missing_spss.png")
    except Exception as e:
        print(f"❌ Erro durante a renderização do template HTML: {e}")

if __name__ == "__main__":
    gerar_relatorio()