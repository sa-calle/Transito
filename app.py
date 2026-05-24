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
# ESTILOS
# ─────────────────────────────────────────────
PALETTE_MAIN   = "#1C1C1E"
PALETTE_ACCENT = "#8A5C3A"
PALETTE_MUTED  = "#8A8070"
PALETTE_CATS   = [
    "#1C1C1E",
    "#8A5C3A",
    "#5C7A5A",
    "#4A6B8A",
    "#8A7040",
    "#6B4A8A",
    "#5A7A78"
]

sns.set(style="whitegrid")

# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────
def apply_academic_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#FAFAF8")
    ax.figure.set_facecolor("#FAFAF8")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#D0CBC2")
    ax.spines["bottom"].set_color("#D0CBC2")

    ax.tick_params(colors="#6B6459", labelsize=9)

    if title:
        ax.set_title(
            title,
            fontsize=11,
            color="#1C1C1E",
            pad=12
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", visible=False)


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos(path):
    return pd.read_excel(path)


# ─────────────────────────────────────────────
# CLASIFICACIÓN VEHICULAR
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


# ─────────────────────────────────────────────
# FACTORES DE DAÑO
# ─────────────────────────────────────────────
FACTORES_DAÑO = {
    "Cat. I — Livianos": 0.0001,
    "Cat. II — Buses": 0.4,
    "Cat. III — Camiones 2 ejes pequeños": 1.14,
    "Cat. IV — Camiones 2 ejes grandes": 2.25,
    "Cat. V — Camiones 3–4 ejes": 3.15,
    "Cat. VI — Camiones 5 ejes": 4.21,
    "Cat. VII — Camiones ≥ 6 ejes": 5.31,
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
# COMPOSICIÓN VEHICULAR (CORREGIDA)
# ─────────────────────────────────────────────
def calcular_composicion(df):
    df_comp = (
        df.groupby("IdCategoriaTarifa")["Trafico"]
        .sum()
        .reset_index()
    )

    df_comp["Tipologia"] = (
        df_comp["IdCategoriaTarifa"]
        .apply(clasificar_categoria)
    )

    df_final = (
        df_comp.groupby("Tipologia")["Trafico"]
        .sum()
        .reset_index()
    )

    # Cálculo corregido sin el atributo erróneo .hbar
    suma_total = df_final["Trafico"].sum()
    df_final["Porcentaje"] = (df_final["Trafico"] / suma_total) * 100 if suma_total > 0 else 0

    K1 = (
        df_final[
            df_final["Tipologia"].isin(CATEGORIAS_PESADAS)
        ]["Porcentaje"]
        .sum()
    )

    return df_final, K1


# ─────────────────────────────────────────────
# FACTOR DE DAÑO
# ─────────────────────────────────────────────
def calcular_FC(df_porcentajes):
    df = df_porcentajes.copy()
    df["f"] = df["Tipologia"].map(FACTORES_DAÑO)

    # Excluir livianos
    df = df[df["Tipologia"] != "Cat. I — Livianos"]

    df["FC_i"] = df["Porcentaje"] * df["f"]

    FC = (
        df["FC_i"].sum()
        / df["Porcentaje"].sum()
    ) if df["Porcentaje"].sum() > 0 else 0

    return FC, df


# ─────────────────────────────────────────────
# CÁLCULO DEL TPD
# ─────────────────────────────────────────────
def calcular_TPD(df):
    df = df.copy()

    # Tráfico total real
    df["Trafico_total"] = (
        df["Trafico"]
        + df["TraficoEvasores"]
        + df["TraficoExentos787"]
    )

    # Manejo de fechas
    df["FechaDesde"] = pd.to_datetime(df["FechaDesde"])
    df["FechaHasta"] = pd.to_datetime(df["FechaHasta"])

    # Días exactos
    df["Dias"] = (
        df["FechaHasta"]
        - df["FechaDesde"]
    ).dt.days + 1

    # Agrupar por periodos
    df_periodo = (
        df.groupby(["FechaDesde", "FechaHasta"])
        .agg({
            "Trafico_total": "sum",
            "Dias": "first"
        })
        .reset_index()
    )

    # TPD del periodo
    df_periodo["TPD"] = (
        df_periodo["Trafico_total"]
        / df_periodo["Dias"]
    )

    df_periodo["Año"] = (
        df_periodo["FechaDesde"].dt.year
    )

    df_periodo["Mes"] = (
        df_periodo["FechaDesde"].dt.month
    )

    # Promedio mensual
    df_periodo = (
        df_periodo.groupby(["Año", "Mes"])["TPD"]
        .mean()
        .reset_index()
    )

    # Matriz mensual
    matriz = df_periodo.pivot_table(
        index="Año",
        columns="Mes",
        values="TPD",
        aggfunc="mean"
    )

    # PROMEDIO NATURAL SIN rellenar NaNs
    TPD = matriz.mean(axis=1).reset_index()
    TPD.columns = ["Año", "TPD"]

    return TPD, matriz


# ─────────────────────────────────────────────
# REGRESIÓN
# ─────────────────────────────────────────────
def regresion_TPD(TPD, año_objetivo):
    x = TPD["Año"].values.reshape(-1, 1)
    y = TPD["TPD"].values

    modelo = LinearRegression()
    modelo.fit(x, y)

    TPD_pred = modelo.predict(np.array([[año_objetivo]]))[0]
    r2 = modelo.score(x, y)

    # --- CÁLCULO DE LA TASA DE CRECIMIENTO ANUAL COMPUESTA (CAGR) ---
    año_inicial = TPD["Año"].min()
    TPD_inicial = modelo.predict(np.array([[año_inicial]]))[0]
    periodo = año_objetivo - año_inicial
    
    if periodo > 0 and TPD_inicial > 0:
        r_estimada = (TPD_pred / TPD_inicial) ** (1 / periodo) - 1
    else:
        r_estimada = 0.0

    return modelo, TPD_pred, r2, r_estimada


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("Análisis de Tránsito")

    try:
        df_raw = cargar_datos("Trafico_Atlantico_2026.xlsx")
    except FileNotFoundError:
        st.error("No se encontró el archivo Excel.")
        st.stop()

    municipios = sorted(
        df_raw["Municipio"]
        .dropna()
        .unique()
        .tolist()
    )

    municipio_sel = st.selectbox("Municipio", municipios)

    años_disponibles = sorted(
        pd.to_datetime(df_raw["FechaDesde"], errors="coerce")
        .dt.year
        .dropna()
        .unique()
        .astype(int)
        .tolist()
    )

    años_excluir = st.multiselect(
        "Años a excluir",
        años_disponibles,
        default=[2020, 2021, 2025, 2026]
    )

    año_pred = st.number_input(
        "Año de proyección",
        min_value=2024,
        max_value=2060,
        value=2030
    )

    K2_input = st.number_input(
        "K2 (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    usar_r_regresion = st.checkbox("Usar 'r' estimada por la regresión", value=True)
    
    if not usar_r_regresion:
        r_input = st.number_input(
            "r (Manual)",
            min_value=0.001,
            max_value=0.20,
            value=0.035,
            step=0.001,
            format="%.3f"
        )
    else:
        st.caption("ℹ️ *La tasa 'r' se calculará automáticamente con la tendencia del modelo.*")
        r_input = 0.035

    n_input = st.number_input(
        "n (años)",
        min_value=1,
        max_value=50,
        value=15
    )


# ─────────────────────────────────────────────
# PREPROCESAMIENTO
# ─────────────────────────────────────────────
df_mun = df_raw[df_raw["Municipio"] == municipio_sel].copy()
df_mun["FechaDesde"] = pd.to_datetime(df_mun["FechaDesde"], errors="coerce")
df_mun["Año_filtro"] = df_mun["FechaDesde"].dt.year

df_filtrado = df_mun[~df_mun["Año_filtro"].isin(años_excluir)].copy()

# Outliers
percentil_99 = df_filtrado["TPDM"].quantile(0.99)
percentil_01 = df_filtrado["TPDM"].quantile(0.01)

df_filtrado = df_filtrado[
    (df_filtrado["TPDM"] >= percentil_01) & 
    (df_filtrado["TPDM"] <= percentil_99)
].copy()

df_filtrado = df_filtrado.drop(columns=["Año_filtro"])


# ─────────────────────────────────────────────
# CÁLCULOS PRINCIPALES
# ─────────────────────────────────────────────
df_composicion, K1 = calcular_composicion(df_filtrado)
FC, df_fc_detalle = calcular_FC(df_composicion)
TPD, matriz_tpd = calcular_TPD(df_filtrado)

modelo, TPD_pred, r2, r_estimada = regresion_TPD(TPD, año_pred)
TPD_diseño = TPD_pred

r_final = r_estimada if usar_r_regresion else r_input

factor_acumulacion = (
    ((1 + r_final) ** n_input - 1)
    / np.log(1 + r_final)
)

N = (
    TPD_diseño
    * (K1 / 100)
    * (K2_input / 100)
    * 365
    * factor_acumulacion
    * FC
)


# ─────────────────────────────────────────────
# TÍTULO Y MÉTRICAS
# ─────────────────────────────────────────────
st.title("Dashboard de Análisis de Tránsito")
st.write(f"Municipio analizado: **{municipio_sel}** | Registros utilizados: **{len(df_filtrado):,}**")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("K1 (%)", f"{K1:.2f}")
col2.metric("FC", f"{FC:.4f}")
col3.metric("TPD Diseño", f"{TPD_diseño:,.0f}")
col4.metric("Precisión ($R^2$)", f"{r2:.4f}")
col5.metric("Tasa Crecimiento ($r$)", f"{r_final * 100:.2f}%")
col6.metric("Ejes Equiv. (N)", f"{N:,.0f}")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "TPD Histórico",
    "Composición Vehicular",
    "Ejes Equivalentes"
])


# ─────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────
with tab1:
    st.subheader("TPD Histórico y Ajuste del Modelo")

    c_reg1, c_reg2 = st.columns(2)
    with c_reg1:
        st.markdown(f"**Coeficiente de Determinación ($R^2$ - Precisión):** `{r2:.4f}`")
    with c_reg2:
        st.markdown(f"**Tasa de Crecimiento Anual de la Tendencia ($r$):** `{r_estimada * 100:.2f}%` o (`{r_estimada:.4f}` decimal)")

    st.dataframe(TPD)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(TPD["Año"], TPD["TPD"], marker="o", color=PALETTE_MAIN)

    X_plot = np.linspace(TPD["Año"].min(), año_pred, 100)
    y_plot = modelo.predict(X_plot.reshape(-1, 1))

    ax.plot(X_plot, y_plot, "--", color=PALETTE_ACCENT)
    ax.scatter(año_pred, TPD_pred, color=PALETTE_ACCENT, s=80)

    apply_academic_style(
        ax,
        title="Proyección del TPD",
        xlabel="Año",
        ylabel="Vehículos/día"
    )
    st.pyplot(fig)


# ─────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Composición Vehicular")
    st.dataframe(df_composicion)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=df_composicion,
        x="Porcentaje",
        y="Tipologia",
        palette=PALETTE_CATS,
        ax=ax
    )

    apply_academic_style(
        ax,
        title="Composición Vehicular",
        xlabel="Porcentaje (%)",
        ylabel=""
    )
    st.pyplot(fig)

    st.subheader("Factor de daño")
    st.dataframe(df_fc_detalle)


# ─────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────
with tab3:
    st.subheader("Ejes Equivalentes")
    st.latex(
        r"""
        N = TPD \cdot \frac{K_1}{100} \cdot \frac{K_2}{100} \cdot 365 \cdot \left( \frac{(1+r)^n -1}{\ln(1+r)} \right) \cdot FC
        """
    )

    st.write(f"### N = {N:,.2f}")
    st.write("---")

    st.write(f"**TPD diseño:** {TPD_diseño:,.2f}")
    st.write(f"**K1 (Vehículos Pesados):** {K1:.2f}%")
    st.write(f"**K2 (Distribución Direccional):** {K2_input:.2f}%")
    st.write(f"**FC (Factor de Daño Global):** {FC:.4f}")
    st.write(f"**r (Tasa Utilizada):** {r_final:.4f} ({r_final * 100:.2f}%)")
    st.write(f"**n (Periodo de diseño):** {n_input} años")