import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------------------
# CONFIGURACIÓN DE LA APP
# ---------------------------
st.set_page_config(page_title="Modelo de Regresión Financiera", layout="wide")

st.title("📊 Modelo de Regresión para Proyecciones Financieras")
st.write("Carga tus datos históricos (CSV o Excel) para calcular correlaciones y regresiones.")

# ---------------------------
# DATASET DE EJEMPLO
# ---------------------------
example_data = pd.DataFrame({
    "Año": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "Ventas": [100, 110, 120, 130, 125, 115, 140, 150, 160],
    "PIB": [2.5, 2.7, 2.9, 2.4, 2.0, -0.5, 3.0, 2.8, 2.5],
    "Desempleo": [5.0, 4.8, 4.6, 4.7, 5.0, 6.5, 5.8, 5.3, 4.9],
    "TipoCambio": [15, 17, 18, 19, 19.5, 22, 21, 20, 18.5],
    "Inflacion": [3.0, 2.8, 3.2, 4.0, 3.6, 4.2, 5.5, 7.0, 6.0]
})

# ---------------------------
# CARGA DE ARCHIVO
# ---------------------------
st.sidebar.header("📂 Cargar archivo")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith("csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
else:
    st.info("Usando dataset de ejemplo. Carga tu archivo en la barra lateral para reemplazarlo.")
    data = example_data.copy()

st.write("### 📋 Datos cargados")
st.dataframe(data)

# ---------------------------
# DETECCIÓN DE VARIABLES
# ---------------------------
all_columns = list(data.columns)

# El usuario elige la variable dependiente
target_var = st.sidebar.selectbox("Selecciona la variable dependiente", all_columns, index=1)

# Las independientes son todas las demás
independent_vars = [v for v in all_columns if v != target_var]

# ---------------------------
# CÁLCULOS DE REGRESIÓN SIMPLE
# ---------------------------
results = []

for var in independent_vars:
    try:
        X = data[[var]].values.reshape(-1, 1)
        y = data[target_var].values

        model = LinearRegression()
        model.fit(X, y)

        beta = model.coef_[0]
        alpha = model.intercept_
        r2 = model.score(X, y)

        results.append({
            "Variable": var,
            "Pendiente (β)": beta,
            "Intersección (α)": alpha,
            "R²": r2
        })
    except Exception as e:
        st.warning(f"No se pudo calcular la regresión con {var}: {e}")

results_df = pd.DataFrame(results)

st.write("### 📈 Resultados de Regresiones Simples")
if not results_df.empty:
    st.dataframe(results_df.style.format({"Pendiente (β)": "{:.3f}", "Intersección (α)": "{:.3f}", "R²": "{:.3f}"}))
else:
    st.warning("No se pudieron calcular regresiones simples. Revisa tu base de datos.")

# ---------------------------
# PRONÓSTICOS MANUALES
# ---------------------------
st.sidebar.header("📊 Pronósticos de variables independientes")

forecast_inputs = {}
for var in independent_vars:
    try:
        default_val = float(data[var].iloc[-1])  # último valor disponible
    except:
        default_val = 0.0
    forecast_inputs[var] = st.sidebar.number_input(f"Pronóstico {var}", value=default_val)

# ---------------------------
# CÁLCULO DE PRONÓSTICOS
# ---------------------------
forecast_results = []
weighted_preds = []
weights =

for res in results:
    var = res["Variable"]
    beta = res["Pendiente (β)"]
    alpha = res["Intersección (α)"]
    r2 = res["R²"]

    x_val = forecast_inputs[var]
    y_pred = alpha + beta * x_val

    forecast_results.append({
        "Variable": var,
        "Pronóstico simple": y_pred,
        "R²": r2
    })

    weighted_preds.append(y_pred * r2)
    weights.append(r2)

# Regresión múltiple ponderada
if sum(weights) > 0:
    y_weighted = sum(weighted_preds) / sum(weights)
else:
    y_weighted = None

forecast_df = pd.DataFrame(forecast_results)

st.write("### 🔮 Pronósticos por Regresión Simple")
if not forecast_df.empty:
    st.dataframe(forecast_df.style.format({"Pronóstico simple": "{:.2f}", "R²": "{:.3f}"}))

st.write("### 📊 Pronóstico Ponderado por R² (Regresión múltiple ponderada)")
if y_weighted is not None:
    st.success(f"Pronóstico final para {target_var}: **{y_weighted:.2f}**")
else:
    st.warning("No fue posible calcular el pronóstico ponderado.")

# ---------------------------
# GRÁFICO
# ---------------------------
if "Año" in data.columns or "Periodo" in data.columns:
    time_col = "Año" if "Año" in data.columns else "Periodo"
    st.write(f"### 📉 Gráfico de {target_var} vs. Pronóstico")
    fig, ax = plt.subplots()
    ax.plot(data[time_col], data[target_var], marker="o", label="Histórico")
    if y_weighted is not None:
        ax.axhline(y=y_weighted, color="red", linestyle="--", label="Pronóstico ponderado")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_var)
    ax.legend()
    st.pyplot(fig)
