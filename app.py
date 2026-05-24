"""
Dashboard de Análisis de Tránsito
Ingeniería Vial — Entorno Académico
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression


# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Tránsito Vial",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# ESTILOS GLOBALES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,300;0,400;0,600;1,300&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Fondo y superficie */
.stApp { background-color: #F7F6F2; }
section[data-testid="stSidebar"] { background-color: #1C1C1E; }
section[data-testid="stSidebar"] * { color: #E8E4DC !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #A09888 !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: #2C2C2E !important;
    border: 1px solid #3A3A3C !important;
    color: #E8E4DC !important;
}

/* Anular h1/h2/h3 genérico de Streamlit */
h1, h2, h3 {
    font-family: 'IBM Plex Serif', serif !important;
    font-weight: 300 !important;
    color: #1C1C1E !important;
}

/* Header principal */
.dash-title {
    font-family: 'IBM Plex Serif', serif;
    font-weight: 300;
    font-size: 2.4rem;
    color: #1C1C1E;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1.2;
}
.dash-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 300;
    font-size: 0.85rem;
    color: #6B6459;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    margin-bottom: 1.8rem;
}
.dash-divider {
    border: none;
    border-top: 2px solid #1C1C1E;
    margin: 0.4rem 0 1.8rem 0;
}

/* Tarjetas de métricas */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E0DBD3;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    text-align: left;
    height: 100%;
}
.metric-card .m-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8A8070;
    margin-bottom: 0.4rem;
}
.metric-card .m-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #1C1C1E;
    line-height: 1;
}
.metric-card .m-unit {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.75rem;
    color: #8A8070;
    margin-top: 0.25rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #D0CBC2;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8A8070;
    padding: 0.65rem 1.4rem;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #1C1C1E !important;
    border-bottom: 2px solid #1C1C1E !important;
    font-weight: 500 !important;
}

/* Sección header */
.section-header {
    font-family: 'IBM Plex Serif', serif;
    font-weight: 400;
    font-size: 1.25rem;
    color: #1C1C1E;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #D0CBC2;
}
.section-note {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.78rem;
    color: #8A8070;
    font-style: italic;
    margin-bottom: 1rem;
}

/* Dataframes */
.stDataFrame { border: 1px solid #E0DBD3 !important; border-radius: 4px; }

/* Sidebar brand / secciones */
.sidebar-brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #5A5450 !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 0.5rem 0 1.2rem 0;
}
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #5A5450 !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 1rem 0 0.3rem 0;
    border-top: 1px solid #2C2C2E;
    margin-top: 0.8rem;
}

/* Formula box */
.formula-box {
    background: #FFFFFF;
    border-left: 3px solid #1C1C1E;
    border-radius: 0 4px 4px 0;
    padding: 1rem 1.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    color: #1C1C1E;
    margin: 1rem 0;
    line-height: 1.7;
}

/* Result highlight (negro) */
.result-highlight {
    background: #1C1C1E;
    color: #F7F6F2;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-highlight .r-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8A8070;
    margin-bottom: 0.4rem;
}
.result-highlight .r-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 500;
    color: #F7F6F2;
    line-height: 1.1;
}
.result-highlight .r-unit {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.8rem;
    color: #8A8070;
    margin-top: 0.3rem;
}

/* Note box */
.note-box {
    background: #FFFFFF;
    border: 1px solid #E0DBD3;
    border-radius: 4px;
    padding: 0.9rem 1.2rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.8rem;
    color: #6B6459;
    margin: 0.8rem 0;
}

/* Expander override */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary,
details > summary {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #1C1C1E !important;
    background-color: #EDEAE4 !important;
    border: 1px solid #D0CBC2 !important;
    border-radius: 4px !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stExpander"] summary:hover,
details > summary:hover {
    background-color: #E0DBD3 !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary * {
    color: #1C1C1E !important;
}

/* KaTeX */
.katex * { color: #1C1C1E !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PALETA Y ESTILO MATPLOTLIB
# ─────────────────────────────────────────────
# ── Paleta principal ──────────────────────────────────────────────────────────
# Carbón base, acento terracota, tierras cálidas + fríos apagados.
# Todos los colores comparten luminosidad baja-media para no romper la estética.
PALETTE_MAIN      = "#1C1C1E"          # carbón casi negro
PALETTE_ACCENT    = "#9C5A32"          # terracota media
PALETTE_ACCENT2   = "#4A7C6F"          # verde-azul pizarra
PALETTE_MUTED     = "#8A8070"          # sepia gris
PALETTE_FILL      = "#C8A882"          # arena cálida (fills suaves)
PALETTE_FILL2     = "#7AABB0"          # azul agua (fills suaves 2)

# Paleta secuencial para categorías — 7 tonos armónicos con la UI
PALETTE_CATS = [
    "#2B2B2E",   # carbón profundo     Cat I
    "#9C5A32",   # terracota           Cat II
    "#4A7C6F",   # verde pizarra       Cat III
    "#6B7FA8",   # azul slate          Cat IV
    "#8A6B3A",   # ocre madera         Cat V
    "#7A5A8A",   # malva oscuro        Cat VI
    "#4A7A72",   # verde agua          Cat VII
]

# Ramp gradiente continuo para barras (de claro a oscuro dentro de cada serie)
import matplotlib.colors as mcolors

def make_bar_colors(n, base="#9C5A32", light_factor=0.55):
    """Genera n tonos interpolados entre una versión clara y la base."""
    base_rgb  = mcolors.to_rgb(base)
    light_rgb = tuple(min(1, c + (1 - c) * light_factor) for c in base_rgb)
    return [mcolors.to_hex(
        tuple(light_rgb[i] + (base_rgb[i] - light_rgb[i]) * t / max(n - 1, 1)
              for i in range(3))
    ) for t in range(n)]


def apply_academic_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#FAFAF8")
    ax.figure.set_facecolor("#FAFAF8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0CBC2")
    ax.spines["bottom"].set_color("#D0CBC2")
    ax.tick_params(colors="#6B6459", labelsize=9)
    ax.xaxis.label.set_color("#6B6459")
    ax.yaxis.label.set_color("#6B6459")
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    if title:
        ax.set_title(title, fontsize=11, fontweight="normal",
                     color="#1C1C1E", fontfamily="serif", pad=12)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E0DBD3", linewidth=0.6, linestyle="--")
    ax.grid(axis="x", visible=False)

def style_legend(ax):
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor("#FFFFFF")
        leg.get_frame().set_edgecolor("#D0CBC2")
        leg.get_frame().set_linewidth(0.8)
        for text in leg.get_texts():
            text.set_color("#4A4035")
            text.set_fontsize(8)


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos(path):
    return pd.read_excel(path)


# ─────────────────────────────────────────────
# CLASIFICACIÓN Y CONSTANTES
# ─────────────────────────────────────────────
def clasificar_categoria(codigo):
    if pd.isna(codigo):
        return "Desconocida"
    codigo = str(codigo).upper()
    if codigo.startswith("VII"):
        return "Cat. VII — Camiones ≥ 6 ejes"
    elif codigo.startswith("VI"):
        return "Cat. VI — Camiones 5 ejes"
    elif codigo.startswith("V") and not codigo.startswith("VI"):
        return "Cat. V — Camiones 3–4 ejes"
    elif codigo.startswith("IV"):
        return "Cat. IV — Camiones 2 ejes grandes"
    elif codigo.startswith("III"):
        return "Cat. III — Camiones 2 ejes pequeños"
    elif codigo.startswith("II"):
        return "Cat. II — Buses"
    elif codigo.startswith("I"):
        return "Cat. I — Livianos"
    else:
        return "Especial / No clasificado"


FACTORES_DAÑO = {
    "Cat. I — Livianos":              0.0001,
    "Cat. II — Buses":                0.4,
    "Cat. III — Camiones 2 ejes pequeños": 1.14,
    "Cat. IV — Camiones 2 ejes grandes":   2.25,
    "Cat. V — Camiones 3–4 ejes":          3.15,
    "Cat. VI — Camiones 5 ejes":           4.21,
    "Cat. VII — Camiones ≥ 6 ejes":        5.31,
}

CATEGORIAS_PESADAS = [
    "Cat. II — Buses",
    "Cat. III — Camiones 2 ejes pequeños",
    "Cat. IV — Camiones 2 ejes grandes",
    "Cat. V — Camiones 3–4 ejes",
    "Cat. VI — Camiones 5 ejes",
    "Cat. VII — Camiones ≥ 6 ejes",
]


# ─────────────────────────────────────────────
# FUNCIONES DE CÁLCULO
# ─────────────────────────────────────────────
def calcular_composicion(df):
    df_comp = df.groupby("IdCategoriaTarifa")["Trafico"].sum().reset_index()
    df_comp["Tipologia"] = df_comp["IdCategoriaTarifa"].apply(clasificar_categoria)
    df_final = df_comp.groupby("Tipologia")["Trafico"].sum().reset_index()
    suma = df_final["Trafico"].sum()
    df_final["Porcentaje"] = (df_final["Trafico"] / suma * 100) if suma > 0 else 0
    K1 = df_final[df_final["Tipologia"].isin(CATEGORIAS_PESADAS)]["Porcentaje"].sum()
    return df_final, K1


def calcular_FC(df_porcentajes):
    df = df_porcentajes.copy()
    df["f"] = df["Tipologia"].map(FACTORES_DAÑO)
    df = df[df["Tipologia"] != "Cat. I — Livianos"]
    df["FC_i"] = df["Porcentaje"] * df["f"]
    FC = df["FC_i"].sum() / df["Porcentaje"].sum() if df["Porcentaje"].sum() > 0 else 0
    return FC, df


def calcular_TPD(df):
    df = df.copy()
    df["Trafico_total"] = (
        df["Trafico"] + df["TraficoEvasores"] + df["TraficoExentos787"]
    )
    df["FechaDesde"] = pd.to_datetime(df["FechaDesde"])
    df["FechaHasta"] = pd.to_datetime(df["FechaHasta"])
    df["Dias"] = (df["FechaHasta"] - df["FechaDesde"]).dt.days + 1

    df_periodo = (
        df.groupby(["FechaDesde", "FechaHasta"])
        .agg({"Trafico_total": "sum", "Dias": "first"})
        .reset_index()
    )
    df_periodo["TPD"] = df_periodo["Trafico_total"] / df_periodo["Dias"]
    df_periodo["Año"] = df_periodo["FechaDesde"].dt.year
    df_periodo["Mes"] = df_periodo["FechaDesde"].dt.month

    df_periodo = (
        df_periodo.groupby(["Año", "Mes"])["TPD"].mean().reset_index()
    )
    matriz = df_periodo.pivot_table(
        index="Año", columns="Mes", values="TPD", aggfunc="mean"
    )
    TPD = matriz.mean(axis=1).reset_index()
    TPD.columns = ["Año", "TPD"]
    return TPD, matriz


def regresion_TPD(TPD, año_objetivo):
    x = TPD["Año"].values.reshape(-1, 1)
    y = TPD["TPD"].values
    modelo = LinearRegression()
    modelo.fit(x, y)
    TPD_pred = modelo.predict(np.array([[año_objetivo]]))[0]
    r2 = modelo.score(x, y)

    año_ini = TPD["Año"].min()
    tpd_ini = modelo.predict(np.array([[año_ini]]))[0]
    periodo  = año_objetivo - año_ini
    r_estimada = (
        (TPD_pred / tpd_ini) ** (1 / periodo) - 1
        if periodo > 0 and tpd_ini > 0 else 0.0
    )
    return modelo, TPD_pred, r2, r_estimada


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">// Ingeniería Vial</div>', unsafe_allow_html=True)
    st.markdown("### Análisis de Tránsito")

    try:
        df_raw = cargar_datos("Trafico_Atlantico_2026.xlsx")
    except FileNotFoundError:
        st.error("Archivo no encontrado: `Trafico_Atlantico_2026.xlsx`")
        st.stop()

    st.markdown('<div class="sidebar-section">Filtro de datos</div>', unsafe_allow_html=True)
    municipios = sorted(df_raw["Municipio"].dropna().unique().tolist())
    municipio_sel = st.selectbox("Municipio", municipios,value="Puerto Colombia")

    años_disponibles = sorted(
        pd.to_datetime(df_raw["FechaDesde"], errors="coerce")
        .dt.year.dropna().unique().astype(int).tolist()
    )
    años_excluir = st.multiselect(
        "Años a excluir",
        años_disponibles,
        default=[y for y in [2020, 2021, 2025, 2026] if y in años_disponibles],
    )

    st.markdown('<div class="sidebar-section">Proyección</div>', unsafe_allow_html=True)
    año_pred = st.number_input("Año de proyección", min_value=2024,
                                max_value=2060, value=2027, step=1)

    st.markdown('<div class="sidebar-section">Parámetros de diseño</div>', unsafe_allow_html=True)
    K2_input = st.number_input("K₂ — Carril de diseño (%)",
                                min_value=0.0, max_value=100.0, value=50.0)

    usar_r_regresion = st.checkbox("Usar r estimada por la regresión", value=True)
    if not usar_r_regresion:
        r_input = st.number_input("r — Tasa de crecimiento (manual)",
                                   min_value=0.001, max_value=0.20,
                                   value=0.035, step=0.001, format="%.3f")
    else:
        r_input = 0.035
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace; font-size:0.68rem; '
            'color:#5A5450; padding:0.3rem 0 0.5rem 0; font-style:italic;">'
            'r se derivará de la tendencia del modelo</div>',
            unsafe_allow_html=True,
        )

    n_input = st.number_input("n — Período de diseño (años)",
                               min_value=1, max_value=50, value=15, step=1)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace; font-size:0.6rem; '
        'color:#5A5450; line-height:1.6;">Metodología AASHTO 93<br>'
        'Factor de daño — Norma INVIAS</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PREPROCESAMIENTO
# ─────────────────────────────────────────────
df_mun = df_raw[df_raw["Municipio"] == municipio_sel].copy()
df_mun["FechaDesde"] = pd.to_datetime(df_mun["FechaDesde"], errors="coerce")
df_mun["_año"] = df_mun["FechaDesde"].dt.year
df_filtrado = df_mun[~df_mun["_año"].isin(años_excluir)].copy()

p99 = df_filtrado["TPDM"].quantile(0.99)
p01 = df_filtrado["TPDM"].quantile(0.01)
df_filtrado = df_filtrado[
    (df_filtrado["TPDM"] >= p01) & (df_filtrado["TPDM"] <= p99)
].copy()
df_filtrado = df_filtrado.drop(columns=["_año"])


# ─────────────────────────────────────────────
# CÁLCULOS PRINCIPALES
# ─────────────────────────────────────────────
df_composicion, K1 = calcular_composicion(df_filtrado)
FC, df_fc_detalle   = calcular_FC(df_composicion)
TPD, matriz_tpd     = calcular_TPD(df_filtrado)

modelo, TPD_pred, r2, r_estimada = regresion_TPD(TPD, año_pred)
TPD_diseño = TPD_pred
r_final    = r_estimada if usar_r_regresion else r_input

factor_acum = ((1 + r_final) ** n_input - 1) / np.log(1 + r_final)
N = TPD_diseño * (K1 / 100) * (K2_input / 100) * 365 * factor_acum * FC


# ─────────────────────────────────────────────
# ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────
año_min = df_filtrado["FechaDesde"].dt.year.min() if not df_filtrado.empty else "—"
año_max = df_filtrado["FechaDesde"].dt.year.max() if not df_filtrado.empty else "—"

st.markdown('<div class="dash-title">Análisis de Tránsito Vial</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="dash-subtitle">Estación: {municipio_sel} &nbsp;·&nbsp; '
    f'Datos: {año_min}–{año_max} &nbsp;·&nbsp; '
    f'N = {len(df_filtrado):,} registros</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)


# ── KPIs globales ──────────────────────────────
exp_N = int(np.floor(np.log10(abs(N)))) if N > 0 else 0
mant_N = N / 10**exp_N

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="m-label">K₁ — Vehículos pesados</div>
        <div class="m-value">{K1:.2f}</div>
        <div class="m-unit">% del flujo total</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="m-label">Factor de daño FC</div>
        <div class="m-value">{FC:.4f}</div>
        <div class="m-unit">ejes equiv. / vehículo</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="m-label">TPD de diseño</div>
        <div class="m-value">{int(TPD_diseño):,}</div>
        <div class="m-unit">vehículos / día</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="m-label">Precisión R²</div>
        <div class="m-value">{r2:.4f}</div>
        <div class="m-unit">bondad del ajuste lineal</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="m-label">Tasa crecimiento r</div>
        <div class="m-value">{r_final*100:.2f}%</div>
        <div class="m-unit">{"estimada por regresión" if usar_r_regresion else "definida manualmente"}</div>
    </div>""", unsafe_allow_html=True)
with c6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="m-label">Ejes equivalentes N</div>
        <div class="m-value">{mant_N:.2f}×10<sup>{exp_N}</sup></div>
        <div class="m-unit">ESALs · {n_input} años</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PESTAÑAS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "1 · TPD Histórico y Proyección",
    "2 · Composición Vehicular",
    "3 · Ejes Equivalentes (N)",
])


# ══════════════════════════════════════════════
# TAB 1 — TPD
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">TPD Histórico y Ajuste del Modelo</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Regresión lineal sobre los años disponibles. '
        'Los años excluidos (pandemia u outliers) no participan en el ajuste.</div>',
        unsafe_allow_html=True,
    )

    # Sub-métricas
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="m-label">TPD proyectado ({año_pred})</div>
            <div class="m-value">{int(TPD_diseño):,}</div>
            <div class="m-unit">vehículos / día</div>
        </div>""", unsafe_allow_html=True)
    with cm2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="m-label">R² del modelo</div>
            <div class="m-value">{r2:.4f}</div>
            <div class="m-unit">coeficiente de determinación</div>
        </div>""", unsafe_allow_html=True)
    with cm3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="m-label">Tasa de crecimiento r</div>
            <div class="m-value">{r_estimada*100:.2f}%</div>
            <div class="m-unit">CAGR · tendencia del modelo</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_g, col_t = st.columns([3, 2])

    with col_g:
        fig, ax = plt.subplots(figsize=(7, 4.2))

        # Área bajo la línea histórica
        ax.fill_between(TPD["Año"], TPD["TPD"],
                        alpha=0.10, color=PALETTE_MAIN, zorder=1)

        # Línea histórica
        ax.plot(TPD["Año"], TPD["TPD"],
                marker="o", markersize=6, color=PALETTE_MAIN,
                linewidth=2, label="TPD histórico",
                markerfacecolor="#FFFFFF", markeredgecolor=PALETTE_MAIN,
                markeredgewidth=1.8, zorder=3)

        # Área de proyección (distinguible del histórico)
        X_plot = np.linspace(TPD["Año"].min(), año_pred, 200)
        y_plot = modelo.predict(X_plot.reshape(-1, 1))
        ax.fill_between(X_plot, y_plot,
                        alpha=0.08, color=PALETTE_ACCENT, zorder=1)
        ax.plot(X_plot, y_plot, "--", color=PALETTE_ACCENT,
                linewidth=1.6, alpha=0.9, label="Tendencia lineal", zorder=2)

        # Punto de proyección con halo
        ax.scatter([año_pred], [TPD_pred], s=130, color=PALETTE_ACCENT,
                   zorder=5, label=f"Proyección {año_pred}",
                   edgecolors="#FFFFFF", linewidths=1.5)
        ax.annotate(
            f"  {int(TPD_pred):,} veh/día",
            xy=(año_pred, TPD_pred),
            xytext=(año_pred, TPD_pred * 1.05),
            fontsize=8, color=PALETTE_ACCENT, fontweight="normal",
            fontfamily="monospace",
        )

        ax.legend(fontsize=8, framealpha=0.95, edgecolor="#D0CBC2",
                  facecolor="#FFFFFF", loc="upper left")
        style_legend(ax)
        apply_academic_style(ax,
            title="Proyección del TPD mediante regresión lineal",
            xlabel="Año", ylabel="TPD (vehículos/día)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_t:
        st.markdown('<div class="section-header" style="font-size:1rem; margin-top:0;">TPD anual</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            TPD.rename(columns={"Año": "Año", "TPD": "TPD (veh/día)"})
               .assign(**{"TPD (veh/día)": lambda d: d["TPD (veh/día)"].round(1)})
               .reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

    st.markdown('<div class="section-header">Matriz de TPD mensual por año</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Promedio mensual de TPD. Las celdas vacías corresponden '
        'a meses sin datos en ese año.</div>',
        unsafe_allow_html=True,
    )
    meses_es = {1:"Ene", 2:"Feb", 3:"Mar", 4:"Abr", 5:"May", 6:"Jun",
                7:"Jul", 8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dic"}
    matriz_display = matriz_tpd.copy().round(0)
    matriz_display.columns = [meses_es.get(c, c) for c in matriz_display.columns]
    st.dataframe(matriz_display, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — Composición
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Composición Vehicular</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Distribución del tráfico por categoría INVIAS. '
        'K₁ representa la fracción de vehículos pesados que contribuyen al deterioro del pavimento.</div>',
        unsafe_allow_html=True,
    )

    ck1, ck2 = st.columns(2)
    with ck1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="m-label">K₁ — Proporción vehículos pesados</div>
            <div class="m-value">{K1:.2f} %</div>
            <div class="m-unit">buses + camiones sobre el flujo total</div>
        </div>""", unsafe_allow_html=True)
    with ck2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="m-label">Factor de daño FC</div>
            <div class="m-value">{FC:.4f}</div>
            <div class="m-unit">ejes equivalentes ponderados por categoría</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_pie, col_bar = st.columns([1, 1])

    with col_pie:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        colors_pie = PALETTE_CATS[: len(df_composicion)]

        # Donut chart (más moderno que pie sólido, menos cargado)
        wedges, texts, autotexts = ax.pie(
            df_composicion["Porcentaje"],
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            colors=colors_pie,
            startangle=90,
            pctdistance=0.80,
            wedgeprops={
                "linewidth": 1.2,
                "edgecolor": "#F7F6F2",
                "width": 0.62,        # donut
            },
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color("#FFFFFF")
            at.set_fontweight("bold")

        # Centro del donut
        ax.text(0, 0, f"K₁\n{K1:.1f}%", ha="center", va="center",
                fontsize=10, color=PALETTE_MAIN, fontfamily="monospace",
                fontweight="bold")

        ax.legend(
            wedges, df_composicion["Tipologia"],
            loc="lower center", bbox_to_anchor=(0.5, -0.22),
            ncol=2, fontsize=7, framealpha=0, edgecolor="none",
            labelcolor="#4A4035",
        )
        ax.set_title("Distribución por categoría vehicular",
                     fontsize=11, fontweight="normal",
                     color="#1C1C1E", fontfamily="serif", pad=10)
        fig.set_facecolor("#FAFAF8")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_bar:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        sorted_df = df_composicion.sort_values("Porcentaje").reset_index(drop=True)
        n_cats = len(sorted_df)
        bar_colors = make_bar_colors(n_cats, base=PALETTE_ACCENT, light_factor=0.55)

        bars = ax.barh(
            sorted_df["Tipologia"], sorted_df["Porcentaje"],
            color=bar_colors, height=0.58,
            edgecolor="#F7F6F2", linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=8,
                     color=PALETTE_MUTED, fontfamily="monospace")
        apply_academic_style(ax,
            title="Porcentaje por tipología",
            xlabel="Porcentaje (%)")
        ax.grid(axis="x", color="#E0DBD3", linewidth=0.6, linestyle="--")
        ax.grid(axis="y", visible=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-header">Tabla de composición</div>', unsafe_allow_html=True)
    df_show = df_composicion[["Tipologia", "Trafico", "Porcentaje"]].copy()
    df_show.columns = ["Tipología", "Tráfico total", "Porcentaje (%)"]
    df_show["Porcentaje (%)"] = df_show["Porcentaje (%)"].round(2)
    df_show["Tráfico total"] = df_show["Tráfico total"].apply(lambda x: f"{int(x):,}")
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Factor de daño por categoría</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">FC pondera la contribución de cada categoría al deterioro '
        'del pavimento según equivalentes de carga estándar (ESAL). '
        'La categoría I (livianos) se excluye del cálculo.</div>',
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    mask = df_fc_detalle["f"].notna()
    sorted_fc = df_fc_detalle[mask].sort_values("f").reset_index(drop=True)
    n_fc = len(sorted_fc)
    fc_colors = make_bar_colors(n_fc, base=PALETTE_ACCENT2, light_factor=0.60)

    bars2 = ax.barh(
        sorted_fc["Tipologia"], sorted_fc["f"],
        color=fc_colors, height=0.55,
        edgecolor="#F7F6F2", linewidth=0.5,
    )
    # Marcador de FC global
    ax.axvline(FC, color=PALETTE_ACCENT, linewidth=1.2, linestyle="--",
               label=f"FC global = {FC:.4f}")
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#D0CBC2", facecolor="#FFFFFF")
    style_legend(ax)

    ax.bar_label(bars2, fmt="%.2f", padding=5, fontsize=8,
                 color=PALETTE_MUTED, fontfamily="monospace")
    apply_academic_style(ax,
        title="Factor unitario de daño por categoría",
        xlabel="Factor (f)")
    ax.grid(axis="x", color="#E0DBD3", linewidth=0.6, linestyle="--")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    df_fc_view = df_fc_detalle[["Tipologia", "Porcentaje", "f", "FC_i"]].copy()
    df_fc_view.columns = ["Tipología", "Porcentaje (%)", "Factor (f)", "FC_i"]
    df_fc_view["Porcentaje (%)"] = df_fc_view["Porcentaje (%)"].round(3)
    df_fc_view["Factor (f)"]     = df_fc_view["Factor (f)"].round(4)
    df_fc_view["FC_i"]           = df_fc_view["FC_i"].round(4)
    st.dataframe(df_fc_view, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3 — Ejes Equivalentes
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Número de Ejes Equivalentes de 8,2 t (N)</div>',
                unsafe_allow_html=True)

    st.latex(
        r"N = TPD \cdot \frac{K_1}{100} \cdot \frac{K_2}{100} \cdot 365 "
        r"\cdot \frac{(1+r)^n - 1}{\ln(1+r)} \cdot FC"
    )

    col_par, col_res = st.columns([1, 1])

    with col_par:
        df_params = pd.DataFrame({
            "Parámetro": ["TPD de diseño", "K₁ — Vehículos pesados",
                          "K₂ — Carril de diseño", "Tasa de crecimiento r",
                          "Período de diseño n", "Factor de daño FC"],
            "Símbolo": ["TPD", "K₁", "K₂", "r", "n", "FC"],
            "Valor": [
                f"{int(TPD_diseño):,} veh/día",
                f"{K1:.3f} %",
                f"{K2_input:.1f} %",
                f"{r_final:.4f} ({r_final*100:.2f} %)",
                f"{int(n_input)} años",
                f"{FC:.4f}",
            ],
        })
        st.dataframe(df_params, use_container_width=True, hide_index=True)

    with col_res:
        st.markdown(f"""
        <div class="result-highlight">
            <div class="r-label">Número de ejes equivalentes acumulados</div>
            <div class="r-value">N = {N:,.0f}</div>
            <div class="r-unit">ESALs · período {n_input} años · carril de diseño</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-top:0.8rem;">
            <div class="m-label">Notación científica</div>
            <div class="m-value">{mant_N:.3f} × 10<sup>{exp_N}</sup></div>
            <div class="m-unit">ejes equivalentes de 8,2 toneladas</div>
        </div>
        """, unsafe_allow_html=True)

    # Curva de sensibilidad a n
    st.markdown('<div class="section-header">Análisis de sensibilidad — Período de diseño</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Variación de N para distintos períodos de diseño, '
        'manteniendo constantes los demás parámetros.</div>',
        unsafe_allow_html=True,
    )

    ns_range = list(range(5, 31))
    Ns_range = [
        TPD_diseño * (K1 / 100) * (K2_input / 100) * 365
        * (((1 + r_final) ** ni - 1) / np.log(1 + r_final)) * FC
        for ni in ns_range
    ]

    fig, ax = plt.subplots(figsize=(9, 3.8))

    Ns_M = [v / 1e6 for v in Ns_range]

    # Área de relleno con gradiente visual (dos capas)
    ax.fill_between(ns_range, Ns_M, alpha=0.07, color=PALETTE_ACCENT2)
    ax.fill_between(ns_range, Ns_M, alpha=0.04, color=PALETTE_MAIN)

    # Línea principal
    ax.plot(ns_range, Ns_M, color=PALETTE_ACCENT2, linewidth=2.2, zorder=3)

    # Línea vertical de diseño
    ax.axvline(n_input, color=PALETTE_ACCENT, linewidth=1.4,
               linestyle="--", alpha=0.9, label=f"n = {n_input} años",  zorder=4)

    # Línea horizontal de N resultante
    ax.axhline(N / 1e6, color=PALETTE_ACCENT, linewidth=0.9,
               linestyle=":", alpha=0.6, zorder=2)

    # Punto de diseño con halo
    ax.scatter([n_input], [N / 1e6], s=110, color=PALETTE_ACCENT,
               edgecolors="#FFFFFF", linewidths=1.8, zorder=6,
               label=f"N = {N/1e6:.2f} M ESALs")

    ax.legend(fontsize=8, framealpha=0.95, edgecolor="#D0CBC2",
              facecolor="#FFFFFF", loc="upper left")
    style_legend(ax)

    apply_academic_style(ax,
        title="Sensibilidad de N respecto al período de diseño",
        xlabel="Período de diseño n (años)",
        ylabel="N (millones de ESALs)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()