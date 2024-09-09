# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_absolute_percentage_error
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.graphics.tsaplots import plot_acf
# 
# # Título de la aplicación
# st.title("Análisis de Datos y Modelado Predictivo")
# 
# # Paso 1: Elegir si el usuario quiere hacer EDA o Modelado
# option = st.radio("¿Qué te gustaría hacer?", ("Análisis Exploratorio de Datos (EDA)", "Modelado Predictivo"))
# 
# # Subida de archivo CSV
# uploaded_file = st.file_uploader("Sube tus datos (CSV)", type="csv")
# 
# if uploaded_file is not None:
#     # Leer el archivo CSV
#     df = pd.read_csv(uploaded_file)
#     st.write("Datos cargados:")
#     st.write(df.head())
# 
#     # Opción 1: Análisis Exploratorio de Datos (EDA)
#     if option == "Análisis Exploratorio de Datos (EDA)":
#         st.write("### Análisis Exploratorio de Datos")
# 
#     # Mostrar los primeros datos
#     st.write("Datos cargados:")
#     st.write(df.head())
# 
#     # Mostrar las dimensiones del dataset
#     st.write("Dimensiones del dataset:")
#     st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
# 
#     # Descripción estadística básica del dataset
#     st.write("Descripción estadística:")
#     st.write(df.describe())
# 
#     # Análisis de valores nulos
#     st.write("Valores nulos por columna:")
#     st.write(df.isnull().sum())
# 
#     # Selección de variables para gráficos
#     column = st.selectbox("Selecciona una columna para análisis gráfico", df.columns)
# 
#     # Histograma
#     st.write(f"Histograma de la columna {column}:")
#     st.bar_chart(df[column])
# 
#     # Desviación estándar
#     st.write(f"Desviación estándar de la columna {column}:")
#     st.write(df[column].std())
# 
#     # Coeficiente de variación
#     if df[column].mean() != 0:
#         coef_var = df[column].std() / df[column].mean()
#         st.write(f"Coeficiente de variación de la columna {column}: {coef_var:.4f}")
#     else:
#         st.write("El coeficiente de variación no se puede calcular (media = 0).")
# 
#     # Gráfico de densidad
#     if st.checkbox("Mostrar gráfico de densidad"):
#         fig, ax = plt.subplots()
#         sns.kdeplot(df[column], ax=ax, fill=True)
#         st.pyplot(fig)
# 
#     # Boxplot para distribución y outliers
#     if st.checkbox("Mostrar Boxplot"):
#         fig, ax = plt.subplots()
#         sns.boxplot(data=df[column], ax=ax)
#         st.pyplot(fig)
# 
#     # Matriz de correlación y heatmap
#     if st.checkbox("Mostrar matriz de correlación"):
#         st.write("Matriz de correlación:")
#         corr_matrix = df.corr()
#         st.write(corr_matrix)
# 
#         st.write("Heatmap de la matriz de correlación:")
#         fig, ax = plt.subplots()
#         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
#         st.pyplot(fig)
# 
#     # Autocorrelación (solo si es una serie temporal o numérica)
#     if st.checkbox("Mostrar autocorrelación"):
#         fig, ax = plt.subplots()
#         plot_acf(df[column].dropna(), ax=ax)
#         st.pyplot(fig)
# 
#     # Gráfico de dispersión interactivo con Plotly
#     if st.checkbox("Mostrar gráfico interactivo"):
#         x_axis = st.selectbox("Selecciona el eje X", df.columns)
#         y_axis = st.selectbox("Selecciona el eje Y", df.columns)
# 
#         fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Gráfico interactivo de {x_axis} vs {y_axis}")
#         st.plotly_chart(fig)
# # Opción 2: Modelado Predictivo
# elif option == "Modelado Predictivo":
#     st.write("### Modelado Predictivo")
# 
#     # Convertir la columna de fecha a tipo datetime si es necesario
#     date_column = st.selectbox("Selecciona la columna de fecha (si aplica)", [None] + df.columns.tolist())
#     if date_column is not None:
#         df[date_column] = pd.to_datetime(df[date_column])
# 
#     # Seleccionar variable objetivo
#     target = st.selectbox("Selecciona la variable objetivo (dependiente)", df.columns)
# 
#     # Seleccionar las variables predictoras
#     features = st.multiselect("Selecciona las variables predictoras (independientes)", df.columns.tolist(), default=df.columns[:-1])
# 
#     if len(features) > 0 and target:
#         X = df[features]
#         y = df[target]
# 
#         # Selección de la observación de prueba (se excluye del entrenamiento)
#         st.write("## Selección de Observación de Prueba")
# 
#         # Si hay una columna de fecha, permitimos la selección por fecha
#         if date_column is not None:
#             test_date = st.selectbox("Selecciona la fecha de prueba", df[date_column].sort_values())
#             test_index = df[date_column].index[df[date_column] == test_date][0]
#         else:
#             test_index = st.slider("Selecciona el índice de la observación de prueba", min_value=0, max_value=len(df)-1, value=len(df)-2)
#             test_date = None
# 
#         # Entrenamiento con los datos anteriores a la observación seleccionada
#         X_train = X[:test_index]  # Datos de entrenamiento
#         y_train = y[:test_index]  # Variable dependiente de entrenamiento
#         X_test = X.iloc[test_index]  # Dato de prueba
#         y_test = y.iloc[test_index]  # Valor real del dato de prueba
# 
#         st.write(f"Entrenando con observaciones anteriores a {test_date or test_index}")
# 
#         # Explicación de modelos disponibles
#         st.write("## Explicación de Modelos")
# 
#         # Expander para modelos de series temporales
#         with st.expander("¿Qué es ARIMA?"):
#             st.write("""
#             ARIMA es un modelo de predicción de series temporales que combina tres componentes:
#             la autoregresión (AR), la diferenciación (I) y el promedio móvil (MA). Es adecuado para datos sin estacionalidad.
#             """)
# 
#         with st.expander("¿Qué es Exponential Smoothing (ETS)?"):
#             st.write("""
#             ETS (Suavización Exponencial) es un modelo de predicción de series temporales que se utiliza para capturar tendencias y estacionalidad
#             en los datos, aplicando un suavizado progresivo que otorga más peso a los datos recientes.
#             """)
# 
#         with st.expander("¿Qué es SARIMA?"):
#             st.write("""
#             SARIMA es una extensión del modelo ARIMA que incorpora componentes estacionales para ajustarse mejor a series temporales
#             con patrones estacionales. Añade parámetros estacionales para capturar la periodicidad de los datos.
#             """)
# 
#         # Expander para modelos tradicionales
#         with st.expander("¿Qué es Regresión Lineal?"):
#             st.write("""
#             La regresión lineal es un modelo de predicción que asume una relación lineal entre la variable dependiente y una o más variables independientes.
#             """)
# 
#         with st.expander("¿Qué es Árbol de Decisión?"):
#             st.write("""
#             Un Árbol de Decisión es un modelo predictivo que utiliza un árbol estructurado para dividir los datos en subconjuntos con decisiones basadas en los valores de las variables predictoras.
#             """)
# 
#         with st.expander("¿Qué es Bosque Aleatorio?"):
#             st.write("""
#             El Bosque Aleatorio es un modelo de conjunto que combina varios árboles de decisión para mejorar la precisión y evitar el sobreajuste.
#             """)
# 
#         with st.expander("¿Qué es K-Nearest Neighbors (KNN)?"):
#             st.write("""
#             KNN es un algoritmo basado en la similitud entre los datos. Para hacer predicciones, utiliza los K vecinos más cercanos a la muestra que se quiere predecir.
#             """)
# 
#         with st.expander("¿Qué es Regresión Polinómica?"):
#             st.write("""
#             La regresión polinómica es una extensión de la regresión lineal que permite modelar relaciones no lineales al incluir términos polinómicos.
#             """)
# 
#         # Seleccionar uno o más modelos de predicción
#         model_choices = st.multiselect("Selecciona uno o más modelos",
#                                        ["Regresión Lineal", "Árbol de Decisión", "Bosque Aleatorio",
#                                         "K-Nearest Neighbors", "Regresión Polinómica",
#                                         "ARIMA", "Exponential Smoothing (ETS)", "SARIMA"])
# 
#         if st.button("Entrenar Modelos"):
#             predictions_dict = {}  # Diccionario para almacenar las predicciones de cada modelo
# 
#             for model_choice in model_choices:
#                 st.write(f"### Entrenando el modelo: {model_choice}")
# 
#                 # Modelos tradicionales
#                 if model_choice == "Regresión Lineal":
#                     model = LinearRegression()
#                     model.fit(X_train, y_train)
#                     prediction = model.predict([X_test])
#                     predictions_dict[model_choice] = prediction
# 
#                 elif model_choice == "Árbol de Decisión":
#                     model = DecisionTreeRegressor()
#                     model.fit(X_train, y_train)
#                     prediction = model.predict([X_test])
#                     predictions_dict[model_choice] = prediction
# 
#                 elif model_choice == "Bosque Aleatorio":
#                     model = RandomForestRegressor()
#                     model.fit(X_train, y_train)
#                     prediction = model.predict([X_test])
#                     predictions_dict[model_choice] = prediction
# 
#                 elif model_choice == "K-Nearest Neighbors":
#                     model = KNeighborsRegressor(n_neighbors=5)
#                     model.fit(X_train, y_train)
#                     prediction = model.predict([X_test])
#                     predictions_dict[model_choice] = prediction
# 
#                 elif model_choice == "Regresión Polinómica":
#                     poly = PolynomialFeatures(degree=2)
#                     X_poly_train = poly.fit_transform(X_train)
#                     X_poly_test = poly.transform([X_test])
#                     model = LinearRegression()
#                     model.fit(X_poly_train, y_train)
#                     prediction = model.predict(X_poly_test)
#                     predictions_dict[model_choice] = prediction
# 
#                 # Modelos de series temporales
#                 elif model_choice == "ARIMA":
#                     p = st.slider("Selecciona el valor de p (autoregresión)", 0, 5, 1)
#                     d = st.slider("Selecciona el valor de d (diferenciación)", 0, 2, 1)
#                     q = st.slider("Selecciona el valor de q (media móvil)", 0, 5, 1)
#                     model_arima = ARIMA(df[target][:test_index], order=(p, d, q))
#                     model_fit = model_arima.fit()
#                     forecast_arima = model_fit.forecast(steps=1)
#                     predictions_dict[model_choice] = forecast_arima
# 
#                 elif model_choice == "Exponential Smoothing (ETS)":
#                     model_ets = ExponentialSmoothing(df[target][:test_index], trend="add", seasonal="add", seasonal_periods=12)
#                     model_fit = model_ets.fit()
#                     forecast_ets = model_fit.forecast(steps=1)
#                     predictions_dict[model_choice] = forecast_ets
# 
#                 elif model_choice == "SARIMA":
#                     p = st.slider("Selecciona el valor de p (autoregresión)", 0, 5, 1)
#                     d = st.slider("Selecciona el valor de d (diferenciación)", 0, 2, 1)
#                     q = st.slider("Selecciona el valor de q (media móvil)", 0, 5, 1)
#                     P = st.slider("Selecciona el valor de P (estacionalidad)", 0, 5, 1)
#                     D = st.slider("Selecciona el valor de D (diferenciación estacional)", 0, 2, 1)
#                     Q = st.slider("Selecciona el valor de Q (media móvil estacional)", 0, 5, 1)
#                     m = st.slider("Selecciona el período estacional (m)", 0, 12, 12)
#                     model_sarima = SARIMAX(df[target][:test_index], order=(p, d, q), seasonal_order=(P, D, Q, m))
#                     model_fit = model_sarima.fit()
#                     forecast_sarima = model_fit.forecast(steps=1)
#                     predictions_dict[model_choice] = forecast_sarima
# 
#             st.write("Modelos entrenados correctamente.")
# 
# 
# #Resultados y análisis gráfico
# 
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# 
# # Mostrar los resultados de los modelos entrenados
# st.write("### Resultados de los Modelos Seleccionados")
# 
# # Listas para almacenar los valores reales y predicciones para cada modelo
# real_values = [y_test]
# model_predictions = []
# 
# # Visualización de los resultados y las métricas de precisión
# for model_choice, prediction in predictions_dict.items():
#     # Cálculo del MAE y MSE
#     mae = mean_absolute_error([y_test], [prediction])
#     mse = mean_squared_error([y_test], [prediction])
# 
#     # Cálculo de la diferencia porcentual
#     diff_percent = abs((y_test - prediction) / y_test) * 100
# 
#     # Mostrar resultados para cada modelo
#     st.write(f"**{model_choice}:**")
#     st.write(f"Valor Real: {y_test}")
#     st.write(f"Predicción: {prediction}")
#     st.write(f"Error Absoluto Medio (MAE): {mae:.4f}")
#     st.write(f"Error Cuadrático Medio (MSE): {mse:.4f}")
#     st.write(f"Diferencia porcentual: {diff_percent:.2f}%")
#     st.write("---")
# 
#     # Agregar predicciones a la lista
#     model_predictions.append(prediction)
# 
# # Gráfico comparativo
# st.write("### Comparación Gráfica de las Predicciones con el Valor Real")
# 
# # Crear un dataframe para gráficos
# df_graph = pd.DataFrame({
#     'Modelo': list(predictions_dict.keys()),
#     'Predicciones': model_predictions,
#     'Valor Real': real_values * len(predictions_dict)  # Replicar el valor real para cada modelo
# })
# 
# # Gráfico de barras para comparar las predicciones con el valor real
# st.bar_chart(df_graph.set_index('Modelo'))
# 
# # Gráfico de dispersión interactivo usando Plotly
# fig = px.scatter(df_graph, x='Modelo', y='Predicciones', labels={'Predicciones':'Predicción'},
#                  title="Predicciones de cada Modelo vs Valor Real")
# fig.add_scatter(x=df_graph['Modelo'], y=df_graph['Valor Real'], mode='lines+markers', name='Valor Real')
# st.plotly_chart(fig)
# 
# # Gráfico de líneas usando Matplotlib para análisis más visual
# st.write("### Gráfico de Líneas - Comparación de Predicciones")
# plt.figure(figsize=(10, 5))
# plt.plot(df_graph['Modelo'], df_graph['Predicciones'], label='Predicciones', marker='o')
# plt.axhline(y=y_test, color='r', linestyle='--', label='Valor Real')
# plt.title('Comparación de Predicciones por Modelo')
# plt.xlabel('Modelo')
# plt.ylabel('Valor Predicho')
# plt.legend()
# st.pyplot(plt)
# 
#     # Sección: Predicción de un valor no conocido
# st.write("### Predicción de un valor no conocido")
# 
# # Seleccionar modelos previamente entrenados
# selected_models = st.multiselect("Selecciona los modelos que deseas usar para predecir", list(predictions_dict.keys()))
# 
# # Verificar si es una serie temporal
# if date_column is not None:
#     # Para series temporales, el usuario introduce una fecha futura
#     st.write("Esta es una serie temporal. Introduce una fecha futura para hacer una predicción.")
#     future_date = st.date_input("Selecciona la fecha que deseas predecir")
# 
#     # Mostrar el futuro valor seleccionado
#     st.write(f"Prediciendo el valor para la fecha: {future_date}")
# 
#     # Hacer predicciones con los modelos seleccionados
#     for model_choice in selected_models:
#         st.write(f"### Predicción con {model_choice}")
# 
#         if model_choice == "ARIMA":
#             # Utilizamos el modelo ARIMA previamente entrenado para predecir el siguiente valor
#             prediction = predictions_dict[model_choice].get_prediction(start=len(df[target]), end=len(df[target]) + 1).predicted_mean[0]
# 
#         elif model_choice == "Exponential Smoothing (ETS)":
#             # Utilizamos el modelo ETS previamente entrenado para predecir
#             prediction = predictions_dict[model_choice].forecast(steps=1)[0]
# 
#         elif model_choice == "SARIMA":
#             # Utilizamos el modelo SARIMA previamente entrenado para predecir
#             prediction = predictions_dict[model_choice].get_forecast(steps=1).predicted_mean[0]
# 
#         st.write(f"Predicción del modelo {model_choice} para la fecha {future_date}: {prediction}")
# 
# else:
#     # Para datos no temporales, el usuario introduce los valores para las variables predictoras
#     st.write("Introduce los valores para las variables predictoras del nuevo dato")
# 
#     # Crear inputs dinámicos para cada una de las variables predictoras
#     new_data = []
#     for feature in features:
#         value = st.number_input(f"Introduce un valor para {feature}")
#         new_data.append(value)
# 
#     # Convertir la entrada en un DataFrame para hacer predicciones
#     new_data = np.array(new_data).reshape(1, -1)
# 
#     # Hacer predicciones con los modelos seleccionados
#     for model_choice in selected_models:
#         st.write(f"### Predicción con {model_choice}")
# 
#         if model_choice == "Regresión Lineal":
#             prediction = predictions_dict[model_choice].predict(new_data)[0]
# 
#         elif model_choice == "Árbol de Decisión":
#             prediction = predictions_dict[model_choice].predict(new_data)[0]
# 
#         elif model_choice == "Bosque Aleatorio":
#             prediction = predictions_dict[model_choice].predict(new_data)[0]
# 
#         elif model_choice == "K-Nearest Neighbors":
#             prediction = predictions_dict[model_choice].predict(new_data)[0]
# 
#         elif model_choice == "Regresión Polinómica":
#             # Si se seleccionó la regresión polinómica, debemos transformar las entradas
#             poly = PolynomialFeatures(degree=2)
#             new_data_poly = poly.fit_transform(new_data)
#             prediction = predictions_dict[model_choice].predict(new_data_poly)[0]
# 
#         st.write(f"Predicción del modelo {model_choice} para el nuevo dato: {prediction}")
