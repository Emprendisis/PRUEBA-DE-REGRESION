# 📈 Pronóstico de Ventas con Modelos de Regresión (Streamlit)

Esta aplicación en **Python + Streamlit** permite pronosticar ventas a partir de variables macroeconómicas (PIB, desempleo, tipo de cambio, inflación, etc.) usando:

- **Correlaciones** con ventas (Pearson).
- **Regresiones lineales simples** para cada variable macro (pendiente β, intersección α, R²).
- **Regresión múltiple ponderada por R²**, donde cada pronóstico simple se pondera con su coeficiente de determinación.

El usuario puede cargar datos históricos en CSV/Excel, introducir pronósticos manualmente o cargar un archivo con pronósticos, y obtener resultados en tablas interactivas y en un archivo Excel descargable.

---

## 🚀 Funcionalidades

- Carga de archivo histórico (CSV/Excel) con **Ventas y variables macro**.
- **Auto-transposición** si el archivo viene en formato invertido.
- Limpieza automática de nombres de columnas.
- Selección de variable dependiente (Ventas) y variables independientes (macros).
- Ingreso de pronósticos:
  - Manualmente en el **sidebar**.
  - O desde otro archivo CSV/Excel.
- Cálculo automático de:
  - Correlaciones con Ventas.
  - Pendiente (β), Intersección (α) y R² para cada regresión simple.
  - Pronóstico por cada regresión simple.
  - Pronóstico ponderado por R² (regresión múltiple ponderada).
- Descarga de resultados en **Excel** con hojas:
  1. Histórico limpio
  2. Correlaciones
  3. Regresiones simples
  4. Pronósticos simples
  5. Pronóstico ponderado

---

## 🛠️ Requisitos

Instalar dependencias:

```bash
pip install streamlit pandas numpy scikit-learn openpyxl xlsxwriter
```

---

## ▶️ Uso local

Ejecutar en la terminal:

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador.

---

## 📂 Archivos del proyecto

- `app.py` → Código principal de la aplicación Streamlit.
- `README.md` → Este archivo de instrucciones.

---

## 📤 Ejemplo de salida

- Tabla de correlaciones con Ventas.
- Tabla de regresiones simples (α, β, R²).
- Tabla con pronósticos de ventas de cada regresión simple.
- Valor final del **pronóstico ponderado por R²**.
- Botón para **descargar Excel** con todos los resultados.

---

✒️ Autor: **Jesús Cedeño**
