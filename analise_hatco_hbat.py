"""
Análise de Outliers e Dados Ausentes — HATCO / HBAT_MISSING
Saída: relatorio_analise.html + Gráficos no Padrão Estatístico SPSS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Any
import datetime
import warnings
from jinja2 import Environment, FileSystemLoader

warnings.filterwarnings("ignore")

# ── Configurações ─────────────────────────────────────────────────────────────

# Caminhos e Nomes de Arquivos
ARQUIVO_HATCO         = "HATCO.csv"               # Base de dados principal da HATCO (geralmente usada para análise de outliers)
ARQUIVO_HBAT          = "HBAT_MISSING.csv"        # Base de dados da HBAT que contém dados ausentes (foco em imputação)
ARQUIVO_HTML          = "relatorio_analise.html"  # Arquivo de saída onde será gerado o relatório final formatado

# Limiares Estatísticos para Detecção de Outliers
LIMIAR_ZSCORE         = 2.5    # Desvios-padrão da média; valores acima disso são considerados outliers (univariados/bivariados)
LIMIAR_PVALOR_CHI2    = 0.001  # Probabilidade limite na curva Chi-Quadrado; abaixo disso, o ponto é um outlier multivariado (Mahalanobis)  

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
        # O scipy.stats.zscore aceita um parâmetro nan_policy='omit'
        # Isso evita termos que dropar os NaNs manualmente antes do cálculo
        z_scores = stats.zscore(df[coluna], nan_policy='omit')
        
        # Filtramos o index onde o valor absoluto do z-score supera o limiar
        outliers_idx = df.index[abs(z_scores) > LIMIAR_ZSCORE]
        resultado[coluna] = outliers_idx.tolist()
        
    return resultado

def detectar_outliers_bivariados(df: pd.DataFrame, col_x: str, col_y: str) -> pd.DataFrame:
    """Detecta outliers bivariados através do Z-score dos resíduos da regressão."""
    dados = df[[col_x, col_y]].dropna()
    
    # Se não houver variação ou dados suficientes, stats.linregress já lida bem
    # ou podemos capturar de forma limpa.
    try:
        slope, intercept, *_ = stats.linregress(dados[col_x], dados[col_y])
        
        # O cálculo do resíduo e seu z-score pode usar a mesma lógica univariada
        valores_previstos = intercept + slope * dados[col_x]
        residuos = dados[col_y] - valores_previstos
        z_residuos = stats.zscore(residuos)
        
        # Filtrando e construindo o DataFrame de retorno de forma direta
        is_outlier = abs(z_residuos) > LIMIAR_ZSCORE
        outliers = dados[is_outlier].copy()
        outliers["residuo_padronizado"] = z_residuos[is_outlier].round(4)
        
        return outliers
    except ValueError:
        # Caso falte dados ou variância para a regressão
        return pd.DataFrame()

def calcular_mahalanobis(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """Calcula a distância de Mahalanobis de forma vetorizada e identifica outliers."""
    dados = df[colunas].dropna()
    gl = len(colunas)
    
    # 1. Validação inicial: precisamos de mais pontos que variáveis
    if len(dados) <= gl:
        return _retornar_df_vazio(df.index, gl)
    
    # 2. Centralização dos dados (X - Média)
    dados_centralizados = dados - dados.mean()
    cov_mat = np.cov(dados.values.T)
    
    try:
        # Usa pseudo-inversa (pinv) direto: ela é estável e já lida com matrizes 
        # quase singulares (multicolinearidade) sem precisar do truque do np.eye
        inv_cov_mat = np.linalg.pinv(cov_mat)
        
        # 3. Cálculo vetorizado de Mahalanobis: D² = diag(X_cent . Inv_Cov . X_cent.T)
        # é infinitamente mais rápido
        d2 = np.sum(dados_centralizados.values @ inv_cov_mat * dados_centralizados.values, axis=1)
        
    except (np.linalg.LinAlgError, ValueError):
        return _retornar_df_vazio(df.index, gl)
    
    # 4. Cálculo Estatístico (Chi-Quadrado)
    pvalor = 1 - stats.chi2.cdf(d2, df=gl)
    
    # 5. Construção do resultado alinhado ao DF original
    resultado_parcial = pd.DataFrame({
        "D2": d2,
        "gl": gl,
        "pvalor": pvalor,
        "outlier": pvalor < LIMIAR_PVALOR_CHI2
    }, index=dados.index)
    
    # Reindex traz de volta as linhas com NaN do DF original preenchendo os padrões
    return resultado_parcial.reindex(df.index).fillna({"outlier": False})


def _retornar_df_vazio(index: pd.Index, gl: int) -> pd.DataFrame:
    """Função auxiliar para manter o código limpo ao retornar estrutura padrão."""
    return pd.DataFrame({
        "D2": np.nan, "gl": gl, "pvalor": np.nan, "outlier": False
    }, index=index)

def imputar_pela_media(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
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
    
    # 1. Mapeamento de colunas (Evita loops repetitivos e melhora a leitura)
    mapa_cols = _mapear_nomes_colunas(df)
    variaveis_escalares_hatco = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x9', 'x10']
    colunas_escalares = [mapa_cols[c] for c in variaveis_escalares_hatco if c in mapa_cols]
    colunas_uni = [mapa_cols[c] for c in ['x1', 'x2', 'x5'] if c in mapa_cols]
    coluna_x, coluna_y = mapa_cols.get('x1'), mapa_cols.get('x2')
    colunas_maha = sorted(colunas_escalares)

    # 2. Execução das análises
    outliers_uni = detectar_outliers_univariados(df, colunas_uni)
    outliers_biv = detectar_outliers_bivariados(df, coluna_x, coluna_y)
    maha = calcular_mahalanobis(df, colunas_maha)
    maha_out = maha[maha["outlier"]] # Filtro booleano direto

    # 3. Consolidação dos índices de outliers (Uso de union/unpacks)
    indices_flagados = set().union(
        outliers_biv.index,
        maha_out.index,
        *[idx for idx in outliers_uni.values()]
    )

    # 4. Purificação dos dados
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

def _filtrar_colunas_criticas_hbat(df: pd.DataFrame) -> List[str]:
    """Filtra colunas numéricas com dados ausentes que sejam menores que V10 (excluindo ID)."""
    colunas_validas = []
    for col in colunas_numericas(df):
        nome_limpo = col.lower()
        if nome_limpo != 'id' and nome_limpo.startswith('v'):
            try:
                # Extrai o número da variável (ex: 'v8' -> 8)
                numero_v = int(nome_limpo.replace('v', ''))
                if numero_v < 10 and df[col].isnull().any():
                    colunas_validas.append(col)
            except ValueError:
                continue
    return colunas_validas

def preparar_contexto_hbat(df: pd.DataFrame) -> Dict[str, Any]:
    """Analisa dados ausentes e aplica imputação pela média na base HBAT."""
    total_casos = len(df)
    
    # 1. Análise de Missing (Aproveitamos operações vetorizadas do Pandas)
    contagem_ausentes = df.isnull().sum()
    percentual_ausentes = (df.isnull().mean() * 100).round(2) # Mais direto que dividir manualmente

    # 2. Filtro de regras específicas (Isolado em função externa para clareza)
    colunas_com_ausente = _filtrar_colunas_criticas_hbat(df)

    # 3. Imputação
    df_imputado = imputar_pela_media(df, colunas_com_ausente)

    # 4. Construção da tabela de resumo de forma pythônica
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
    """Função principal que gera o relatório completo."""
    df_hatco = carregar_csv(ARQUIVO_HATCO)
    df_hbat = carregar_csv(ARQUIVO_HBAT)

    ctx_hatco = preparar_contexto_hatco(df_hatco)
    ctx_hbat = preparar_contexto_hbat(df_hbat)

    # Geração dos gráficos
    gerar_boxplot(df_hatco, ctx_hatco["colunas_uni"])
    gerar_dispersao(df_hatco, ctx_hatco["coluna_x"], ctx_hatco["coluna_y"])
    gerar_qq_mahalanobis(ctx_hatco["maha"], ctx_hatco["colunas_maha"])
    gerar_grafico_missing(ctx_hbat["percentual_ausentes"])

    # Preparação do contexto completo para o template
    contexto = {
        "titulo": "Diagnóstico Multivariado de Dados — Padrão SPSS",
        "limiar_zscore": LIMIAR_ZSCORE,
        "limiar_pvalor": LIMIAR_PVALOR_CHI2,
        "data_geracao": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        **ctx_hatco,
        **ctx_hbat
    }

    # Renderização com Jinja2
    env = Environment(loader=FileSystemLoader('.'), autoescape=True)
    template = env.get_template('template_relatorio.html')
    contexto['df_hatco'] = df_hatco
    contexto['df_hbat']  = df_hbat
    html = template.render(contexto)

    with open(ARQUIVO_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Relatório gerado com sucesso: {ARQUIVO_HTML}")
    print("Gráficos gerados: boxplot_spss.png, dispersao_spss.png, qqplot_spss.png, missing_spss.png")

if __name__ == "__main__":
    gerar_relatorio()