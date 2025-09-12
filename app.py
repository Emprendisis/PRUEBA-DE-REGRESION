
# app.py
# Streamlit app: Pronóstico de Ventas con Regresiones (simples y ponderada por R²)
# Requisitos: streamlit, pandas, numpy, scikit-learn, openpyxl o xlsxwriter
# Ejecuta local: streamlit run app.py

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Pronóstico Ventas – Regresión", layout="wide")

# ---------- Utilidades ----------

def load_dataframe(uploaded_file):
    """Carga CSV o Excel, devolviendo un DataFrame."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        # Intentar leer la primera hoja
        return pd.read_excel(uploaded_file, sheet_name=0)
    else:
        st.error("Formato no soportado. Sube CSV o Excel.")
        return None

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia nombres de columnas: quita espacios, caracteres raros, uniformiza."""
    def _norm(c):
        if not isinstance(c, str):
            c = str(c)
        c = c.strip()
        c = c.replace("%", "pct")
        c = c.replace(" ", "_")
        c = c.replace("-", "_")
        c = c.replace("/", "_")
        return c
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df

def maybe_transpose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heurística: si transponer aumenta el número de columnas numéricas útiles, se transpone.
    Además, si después de transponer hay más coincidencias de nombres típicos, se transpone.
    """
    def score(df_):
        # columnas numéricas
        num_cols = df_.select_dtypes(include=[np.number]).shape[1]
        # coincidencias con nombres frecuentes
        cols_lower = [str(c).lower() for c in df_.columns]
        common = ["venta", "ventas", "pib", "desempleo", "inflacion", "inflación", "tipo", "cambio", "tc", "usd_mxn"]
        hits = sum(any(word in c for word in common) for c in cols_lower)
        return num_cols + hits

    s1 = score(df)
    s2 = score(df.T)
    return df.T if s2 > s1 else df

def ensure_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convierte a numéricas las columnas indicadas, silenciando errores."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pearson_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calcula R² de forma segura."""
    if y is None or y_hat is None:
        return np.nan
    if len(y) != len(y_hat) or len(y) < 2:
        return np.nan
    ss_res = np.nansum((y - y_hat)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res/ss_tot

def fit_simple_regression(x: pd.Series, y: pd.Series):
    """Ajusta y = alpha + beta*x. Devuelve alpha, beta, r2 y el modelo."""
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 2:
        return np.nan, np.nan, np.nan, None
    X = df.iloc[:, 0].values.reshape(-1, 1)
    Y = df.iloc[:, 1].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, Y)
    beta = float(model.coef_[0][0])
    alpha = float(model.intercept_[0])
    y_hat = model.predict(X).flatten()
    r2 = pearson_r2(df.iloc[:, 1].values, y_hat)
    return alpha, beta, r2, model

def weighted_forecast_by_r2(simple_preds: dict) -> float:
    """Promedio ponderado por R² de los pronósticos individuales (ignora NaN o R²<=0)."""
    num = 0.0
    den = 0.0
    for k, rec in simple_preds.items():
        y_hat = rec.get("forecast")
        r2 = rec.get("r2")
        if y_hat is None or pd.isna(y_hat) or r2 is None or pd.isna(r2):
            continue
        if r2 <= 0:
            continue
        num += y_hat * r2
        den += r2
    if den == 0:
        return np.nan
    return num / den

def to_excel_bytes(dfs: dict) -> bytes:
    """Recibe un dict {nombre_hoja: DataFrame} y devuelve un archivo Excel en bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet, df in dfs.items():
            # limitar nombre hoja a 31 chars
            safe_sheet = str(sheet)[:31] if sheet else "Sheet1"
            df.to_excel(writer, index=False, sheet_name=safe_sheet)
    return output.getvalue()

# ---------- UI ----------
st.title("📈 Pronóstico de Ventas – Regresiones (Streamlit)")
st.caption("Carga datos históricos, ingresa pronósticos y obtén correlaciones, regresiones simples y pronóstico ponderado por R².")

with st.sidebar:
    st.header("1) Datos históricos")
    file_hist = st.file_uploader("Sube el histórico (CSV o Excel)", type=["csv", "xlsx", "xls"])
    auto_transpose = st.checkbox("Detectar y transponer automáticamente si hace falta", value=True)
    st.markdown("---")
    st.header("2) Pronósticos")
    forecast_mode = st.radio("¿Cómo quieres ingresar los pronósticos?", ["Manual (sidebar)", "Cargar archivo de pronósticos"])

df_hist = None
if file_hist is not None:
    try:
        df_hist = load_dataframe(file_hist)
        if df_hist is not None:
            df_hist = clean_columns(df_hist)
            if auto_transpose:
                df_hist = maybe_transpose(df_hist)
            # Mostrar una vista previa
            st.subheader("Vista previa – Datos históricos (después de limpieza)")
            st.dataframe(df_hist.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Error al cargar el archivo histórico: {e}")

if df_hist is not None and df_hist.shape[1] >= 2:
    # Selección de columnas
    st.markdown("### Selecciona columnas")
    default_sales = None
    # Heurísticas para ventas
    for c in df_hist.columns:
        lc = str(c).lower()
        if "venta" in lc:  # ventas, venta
            default_sales = c
            break
    sales_col = st.selectbox("Columna de Ventas (variable dependiente)", options=list(df_hist.columns), index=(list(df_hist.columns).index(default_sales) if default_sales in df_hist.columns else 0))

    # Preseleccionar macros típicas
    candidate_macros = [c for c in df_hist.columns if c != sales_col]
    preselect = []
    for c in candidate_macros:
        lc = str(c).lower()
        if any(w in lc for w in ["pib", "desemple", "inflac", "tipo_cambio", "tc", "usd", "mxn", "dolar", "dólar", "cambio", "fx", "usd_mxn", "usdmxn", "tipo"]):
            preselect.append(c)
    macro_cols = st.multiselect("Columnas macro (variables independientes)", options=candidate_macros, default=(preselect if preselect else candidate_macros))

    # Asegurar numéricas
    df_num = ensure_numeric(df_hist, [sales_col] + macro_cols)

    # -------- Correlaciones --------
    corr_rows = []
    for m in macro_cols:
        c = pd.concat([df_num[sales_col], df_num[m]], axis=1).dropna()
        val = c.corr(method="pearson").iloc[0, 1] if c.shape[0] >= 2 else np.nan
        corr_rows.append({"Variable": m, "Correlación_Ventas": val})
    df_corr = pd.DataFrame(corr_rows).sort_values("Correlación_Ventas", ascending=False)

    st.markdown("### Correlaciones con Ventas (Pearson)")
    st.dataframe(df_corr, use_container_width=True)

    # -------- Regresiones simples --------
    reg_rows = []
    simple_models = {}
    for m in macro_cols:
        alpha, beta, r2, model = fit_simple_regression(df_num[m], df_num[sales_col])
        reg_rows.append({"Variable": m, "Alpha (α)": alpha, "Beta (β)": beta, "R²": r2})
        simple_models[m] = {"alpha": alpha, "beta": beta, "r2": r2, "model": model}
    df_regs = pd.DataFrame(reg_rows).sort_values("R²", ascending=False)

    st.markdown("### Regresiones simples (y = α + β·x)")
    st.dataframe(df_regs, use_container_width=True)

    # -------- Pronósticos --------
    st.markdown("### Pronósticos de variables macro")
    forecasts = {}

    if forecast_mode == "Manual (sidebar)":
        with st.sidebar:
            st.subheader("Pronósticos manuales")
            for m in macro_cols:
                # Valor por defecto: último valor no nulo del histórico
                last_val = pd.to_numeric(df_num[m], errors="coerce").dropna()
                default_val = float(last_val.iloc[-1]) if not last_val.empty else 0.0
                forecasts[m] = st.number_input(f"Pronóstico {m}", value=default_val, format="%.6f")
    else:
        with st.sidebar:
            st.subheader("Cargar archivo de pronósticos (CSV/Excel)")
            file_fore = st.file_uploader("Archivo de pronósticos", type=["csv", "xlsx", "xls"], key="forefile")
            period_col_name = st.text_input("Nombre de la columna de período (opcional)", value="periodo")
            period_value = st.text_input("Filtrar por período (opcional)", value="")
        df_fore = None
        if file_fore is not None:
            try:
                df_fore = load_dataframe(file_fore)
                if df_fore is not None:
                    df_fore = clean_columns(df_fore)
                    st.caption("Vista previa – Pronósticos (limpios)")
                    st.dataframe(df_fore.head(20), use_container_width=True)
                    # Intentar mapear columnas
                    df_fore_num = ensure_numeric(df_fore, macro_cols)
                    # Si el usuario especificó un período, filtrar
                    if period_col_name and period_col_name in df_fore_num.columns and period_value:
                        filt = df_fore_num[df_fore_num[period_col_name].astype(str) == str(period_value)]
                        row = filt.tail(1)
                    else:
                        # tomar la última fila completa como pronóstico
                        row = df_fore_num.tail(1)
                    for m in macro_cols:
                        if m in row.columns and not row[m].isna().all():
                            forecasts[m] = float(row[m].iloc[0])
                        else:
                            # fallback al último dato histórico
                            last_val = pd.to_numeric(df_num[m], errors="coerce").dropna()
                            forecasts[m] = float(last_val.iloc[-1]) if not last_val.empty else np.nan
            except Exception as e:
                st.error(f"Error al cargar pronósticos: {e}")

        # si no se subió archivo, aún permitir inputs manuales por defecto
        if file_fore is None:
            with st.sidebar:
                st.info("No se detectó archivo de pronósticos. Ingresa manualmente:")
                for m in macro_cols:
                    last_val = pd.to_numeric(df_num[m], errors="coerce").dropna()
                    default_val = float(last_val.iloc[-1]) if not last_val.empty else 0.0
                    forecasts[m] = st.number_input(f"Pronóstico {m}", value=default_val, format="%.6f", key=f"manual_{m}")

    # -------- Pronóstico por regresión simple de cada variable --------
    simple_preds = {}
    pred_rows = []
    for m in macro_cols:
        alpha = simple_models[m]["alpha"]
        beta = simple_models[m]["beta"]
        r2 = simple_models[m]["r2"]
        x_fore = forecasts.get(m, np.nan)
        if not pd.isna(alpha) and not pd.isna(beta) and not pd.isna(x_fore):
            y_hat = alpha + beta * x_fore
        else:
            y_hat = np.nan
        simple_preds[m] = {"forecast": y_hat, "r2": r2}
        pred_rows.append({"Variable": m, "Pronóstico_X": x_fore, "Venta_Pronosticada (simple)": y_hat, "R²": r2})
    df_pred_simples = pd.DataFrame(pred_rows).sort_values("R²", ascending=False)

    st.markdown("### Pronósticos de Ventas por regresiones simples")
    st.dataframe(df_pred_simples, use_container_width=True)

    # -------- Pronóstico ponderado por R² --------
    y_hat_weighted = weighted_forecast_by_r2(simple_preds)
    st.markdown("### Pronóstico de Ventas – Regresión múltiple ponderada por R²")
    c1, c2 = st.columns([1,3])
    with c1:
        st.metric("Venta Pronosticada (ponderada)", f"{y_hat_weighted:,.4f}" if not pd.isna(y_hat_weighted) else "N/D")

    # -------- Descarga en Excel --------
    st.markdown("### Descargar resultados")
    dfs_out = {
        "01_Historico_limpio": df_num.reset_index(drop=True),
        "02_Correlaciones": df_corr.reset_index(drop=True),
        "03_Regresiones_simples": df_regs.reset_index(drop=True),
        "04_Pronosticos_simples": df_pred_simples.reset_index(drop=True),
        "05_Pronostico_ponderado": pd.DataFrame([{"Venta_Pronosticada_ponderada_R2": y_hat_weighted}]),
    }
    excel_bytes = to_excel_bytes(dfs_out)
    st.download_button(
        label="⬇️ Descargar Excel con resultados",
        data=excel_bytes,
        file_name="pronostico_regresion_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Carga un archivo histórico válido para continuar. Debe incluir la columna de Ventas y al menos una variable macro (PIB, desempleo, tipo de cambio %, inflación, etc.).")
