# Academic Performance Predictor (Streamlit)

Aplicación interactiva de predicción del rendimiento académico basada en el proyecto de Google Colab.

## Publicar en Streamlit Community Cloud

1. Crea un repositorio en GitHub.
2. Sube `app.py`, `requirements.txt` y `README.md` (no necesitas subir datos personales).
3. En https://share.streamlit.io elige **Create app** y conecta tu repositorio.
4. Establece `app.py` como archivo principal y pulsa **Deploy**.
5. En la aplicación, sube el Excel de respuestas de Google Forms en el panel izquierdo.

## Ejecutar en una computadora

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Qué incluye

- Carga de Excel y asignación flexible de las cuatro variables.
- Resumen, tablas y gráficas de la encuesta.
- Correlación de Pearson, Chi-cuadrada, ANOVA y Mann-Whitney U.
- Regresión lineal múltiple y predicción de calificación.

**Nota:** El archivo Excel no está incluido. Para reproducir los resultados de tu proyecto necesitas subir el mismo Excel utilizado en Colab. El modelo se ajusta a todas las filas válidas, sin evaluación independiente.
