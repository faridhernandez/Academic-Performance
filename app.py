"""Academic Performance Predictor — aplicación Streamlit del análisis en Colab."""
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Academic Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Academic Performance Predictor")
st.caption("Análisis de hábitos académicos y predicción de calificaciones")


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def suggest(columns, variable):
    rules = {
        "study": [("study hours",), ("hours per week", "stud"), ("studying", "hours")],
        "sleep": [("sleep hours",), ("hours of sleep",), ("sleep", "night")],
        "stress": [("stress level",), ("stress", "scale of 1"), ("stress", "very low")],
        "grade": [("midterm grade",), ("midterm", "grade"), ("current grade",)],
    }
    for keywords in rules[variable]:
        for column in columns:
            if all(word in normalize(column) for word in keywords):
                return column
    return None


def outcome(p, alpha):
    return "Rechazamos H₀" if p < alpha else "No rechazamos H₀"


def test_metrics(stat_name, stat, p, alpha):
    a, b, c = st.columns(3)
    a.metric(stat_name, f"{stat:.4f}")
    b.metric("p-value", f"{p:.4f}")
    c.metric("Decisión", outcome(p, alpha))


st.sidebar.title("Tu encuesta")
upload = st.sidebar.file_uploader("Carga el Excel de Google Forms", type=["xlsx"])
if upload is None:
    st.info("Para comenzar, carga el Excel con las respuestas de tu encuesta desde el panel izquierdo.")
    st.markdown("### ¿Qué permite esta aplicación?")
    st.write("**Análisis:** Pearson, Chi-cuadrada, ANOVA, Mann-Whitney U y gráficas.")
    st.write("**Predicción:** regresión lineal según horas de estudio, sueño y nivel de estrés.")
    st.stop()

try:
    raw = pd.read_excel(upload)
    raw.columns = [str(c).strip() for c in raw.columns]
except Exception as err:
    st.error(f"No se pudo leer el Excel: {err}")
    st.stop()

columns = raw.columns.tolist()
if len(columns) < 4:
    st.error("El archivo necesita por lo menos cuatro columnas.")
    st.stop()

st.sidebar.subheader("Variables")
with st.sidebar.expander("Elegir columnas", expanded=False):
    st.caption("Revisa que las preguntas queden vinculadas a sus variables correctas.")
    assignments = {}
    for variable, label in [
        ("study", "Horas de estudio por semana"),
        ("sleep", "Horas de sueño por noche"),
        ("stress", "Nivel de estrés (1 a 5)"),
        ("grade", "Calificación de exámenes parciales"),
    ]:
        proposed = suggest(columns, variable)
        assignments[variable] = st.selectbox(
            label, columns, index=columns.index(proposed) if proposed in columns else 0
        )
if len(set(assignments.values())) < 4:
    st.error("Selecciona cuatro columnas distintas en el panel lateral.")
    st.stop()

df = pd.DataFrame({
    "Study_Hours": pd.to_numeric(raw[assignments["study"]], errors="coerce"),
    "Sleep_Hours": pd.to_numeric(raw[assignments["sleep"]], errors="coerce"),
    "Stress_Level": pd.to_numeric(raw[assignments["stress"]], errors="coerce"),
    "Midterm_Grade": pd.to_numeric(raw[assignments["grade"]], errors="coerce"),
}).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
if df.empty:
    st.error("No se encontraron respuestas numéricas válidas. Revisa las columnas seleccionadas.")
    st.stop()

alpha = st.sidebar.selectbox("Nivel de significancia", [0.05, 0.01, 0.10], format_func=lambda x: f"{x:.0%}")
st.sidebar.caption(f"Respuestas válidas: {len(df)} de {len(raw)}")

features = ["Study_Hours", "Sleep_Hours", "Stress_Level"]
model = None
if len(df) >= 2:
    model = LinearRegression().fit(df[features], df["Midterm_Grade"])

tabs = st.tabs(["📊 Resumen", "🗂️ Datos", "🧪 Pruebas estadísticas", "🎯 Predictor"])

with tabs[0]:
    st.subheader("Resumen de las respuestas")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estudiantes", len(df))
    c2.metric("Calificación media", f"{df['Midterm_Grade'].mean():.2f}/10")
    c3.metric("Estudio semanal", f"{df['Study_Hours'].mean():.1f} h")
    c4.metric("Sueño nocturno", f"{df['Sleep_Hours'].mean():.1f} h")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="Midterm_Grade", nbins=10, title="Distribución de calificaciones")
        fig.update_layout(xaxis_title="Calificación", yaxis_title="Estudiantes")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(df, x="Stress_Level", y="Midterm_Grade", color="Sleep_Hours",
                         title="Estrés y calificaciones", hover_data=["Study_Hours"])
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Datos utilizados")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.subheader("Estadística descriptiva")
    st.dataframe(df.describe().T.round(3), use_container_width=True)
    st.caption(f"Filas excluidas por datos incompletos o no numéricos: {len(raw)-len(df)}")

with tabs[2]:
    st.subheader("Pruebas estadísticas")
    st.caption(f"Se utiliza α = {alpha:.2f}. Las pruebas reproducen los procedimientos del proyecto de Colab.")
    with st.container(border=True):
        st.markdown("### 1. Correlación de Pearson")
        st.write("H₀: no existe correlación entre estrés y calificación. H₁: la correlación es negativa.")
        if len(df) >= 3 and df["Stress_Level"].nunique() > 1 and df["Midterm_Grade"].nunique() > 1:
            r, p = stats.pearsonr(df["Stress_Level"], df["Midterm_Grade"])
            test_metrics("r", r, p, alpha)
            st.write("Se detecta una relación negativa significativa." if p < alpha and r < 0
                     else "No hay evidencia de una relación negativa significativa con la prueba presentada.")
            st.caption("Como el Colab original, se muestra el p-value bilateral de Pearson.")
        else:
            st.warning("Se necesitan al menos tres observaciones con variación en estrés y calificación.")

    with st.container(border=True):
        st.markdown("### 2. Chi-cuadrada de independencia")
        st.write("H₀: estrés y horas de sueño son independientes. H₁: existe asociación.")
        test_df = df.copy()
        test_df["Stress_Cat"] = np.where(test_df.Stress_Level >= 4, "High Stress", "Low/Med Stress")
        test_df["Sleep_Cat"] = np.where(test_df.Sleep_Hours < 6, "Low Sleep (<6h)", "Sufficient Sleep (>=6h)")
        ct = pd.crosstab(test_df.Stress_Cat, test_df.Sleep_Cat)
        st.dataframe(ct, use_container_width=True)
        if ct.shape == (2, 2):
            try:
                chi2, p, dof, expected = stats.chi2_contingency(ct)
                test_metrics("Chi-cuadrada", chi2, p, alpha)
                st.write("Hay evidencia de asociación entre las variables." if p < alpha
                         else "No hay evidencia suficiente de asociación entre las variables.")
                if np.any(expected < 5):
                    st.warning("Hay frecuencias esperadas menores a 5; interpreta con cautela.")
            except ValueError as err:
                st.warning(f"No es posible calcular la prueba: {err}")
        else:
            st.warning("Se requieren casos en ambas categorías de estrés y sueño.")

    with st.container(border=True):
        st.markdown("### 3. ANOVA de un factor")
        st.write("H₀: la calificación promedio es igual en todos los niveles de estrés.")
        groups = [group.Midterm_Grade.to_numpy() for _, group in df.groupby("Stress_Level")]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
            if np.isfinite(p):
                test_metrics("F", f, p, alpha)
                st.write("Al menos una media difiere." if p < alpha
                         else "No se encontró evidencia suficiente de diferencias entre medias.")
            else:
                st.warning("ANOVA no pudo determinar un resultado válido con estos datos.")
        else:
            st.warning("Se requieren al menos dos niveles de estrés.")

    with st.container(border=True):
        st.markdown("### 4. Mann-Whitney U")
        st.write("H₀: los dos grupos de sueño tienen la misma distribución de calificaciones.")
        low = df.loc[df.Sleep_Hours < 6, "Midterm_Grade"]
        high = df.loc[df.Sleep_Hours >= 6, "Midterm_Grade"]
        st.caption(f"Menos de 6 h: {len(low)} estudiantes; 6 h o más: {len(high)} estudiantes")
        if len(low) > 0 and len(high) > 0:
            u, p = stats.mannwhitneyu(low, high, alternative="two-sided")
            test_metrics("U", u, p, alpha)
            st.write("Hay evidencia de diferencias entre los grupos." if p < alpha
                     else "No hay evidencia suficiente de diferencias entre los grupos.")
        else:
            st.warning("Se necesitan observaciones en los dos grupos de sueño.")

with tabs[3]:
    st.subheader("Predice tu calificación")
    st.write("Ingresa tus hábitos para estimar tu calificación promedio de parcial sobre 10.")
    if model is None:
        st.warning("Se necesitan al menos dos respuestas para entrenar el modelo.")
    else:
        with st.form("prediction_form"):
            a, b, c = st.columns(3)
            with a:
                study = st.number_input("Horas de estudio por semana", 0.0, 80.0, 8.0, 0.5)
            with b:
                sleep = st.number_input("Horas de sueño por noche", 0.0, 24.0, 8.0, 0.5)
            with c:
                stress = st.slider("Nivel de estrés (1 = bajo, 5 = alto)", 1, 5, 3)
            submit = st.form_submit_button("🎯 Predecir calificación", use_container_width=True)
        if submit:
            person = pd.DataFrame({"Study_Hours": [study], "Sleep_Hours": [sleep], "Stress_Level": [stress]})
            estimated = float(np.clip(model.predict(person)[0], 0, 10))
            st.metric("Calificación estimada", f"{estimated:.2f} / 10")
            st.progress(estimated / 10)
            st.subheader("Consejos sobre tus hábitos")
            advice = []
            if stress >= 4:
                advice.append("⚠️ Estrés alto: intenta organizar tus actividades y tomar descansos.")
            if study < 3:
                advice.append("📚 Estudio: considera aumentar el tiempo de preparación.")
            if sleep < 6:
                advice.append("😴 Sueño: dormir menos de 6 horas puede afectar la concentración.")
            for msg in advice or ["✅ Tus hábitos tienen un buen equilibrio según los rangos del proyecto."]:
                st.write(msg)
            st.caption("Es una estimación basada en una encuesta, no una garantía de calificación.")
        st.markdown("### Coeficientes de regresión")
        coefficients = pd.DataFrame({"Variable": features, "Coeficiente": model.coef_})
        st.dataframe(coefficients.round(3), hide_index=True, use_container_width=True)
        st.caption(f"Intercepto: {model.intercept_:.2f}. La regresión se entrena con todos los registros válidos.")
