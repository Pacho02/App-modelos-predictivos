import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf

# Título de la aplicación
st.title("Análisis de Datos y Modelado Predictivo")

# Paso 1: Elegir si el usuario quiere hacer EDA o Modelado
option = st.radio("¿Qué te gustaría hacer?", ("Análisis Exploratorio de Datos (EDA)", "Modelado Predictivo"))

# Aquí asegúrate de que date_column está definido al principio
date_column = None

# Subida de archivo CSV o Excel
uploaded_file = st.file_uploader("Sube tus datos (CSV o Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Leer el archivo dependiendo de su formato
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    st.write("Datos cargados:")
    st.write(df.head())

    # Opción 1: Análisis Exploratorio de Datos (EDA)
    if option == "Análisis Exploratorio de Datos (EDA)":
        st.write("### Análisis Exploratorio de Datos")

        # Mostrar los primeros datos
        st.write("Datos cargados:")
        st.write(df.head())

        # Mostrar las dimensiones del dataset
        st.write("Dimensiones del dataset:")
        st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

        # Descripción estadística básica del dataset
        st.write("Descripción estadística:")
        st.write(df.describe())

        # Análisis de valores nulos
        st.write("Valores nulos por columna:")
        st.write(df.isnull().sum())

        # Selección de variables para gráficos
        column = st.selectbox("Selecciona una columna para análisis gráfico", df.columns)

        # Histograma
        st.write(f"Histograma de la columna {column}:")
        st.bar_chart(df[column])

        # Desviación estándar
        st.write(f"Desviación estándar de la columna {column}:")
        st.write(df[column].std())

        # Coeficiente de variación
        if df[column].mean() != 0:
            coef_var = df[column].std() / df[column].mean()
            st.write(f"Coeficiente de variación de la columna {column}: {coef_var:.4f}")
        else:
            st.write("El coeficiente de variación no se puede calcular (media = 0).")

        # Gráfico de densidad
        if st.checkbox("Mostrar gráfico de densidad"):
            fig, ax = plt.subplots()
            sns.kdeplot(df[column], ax=ax, fill=True)
            st.pyplot(fig)

        # Boxplot para distribución y outliers
        if st.checkbox("Mostrar Boxplot"):
            fig, ax = plt.subplots()
            sns.boxplot(data=df[column], ax=ax)
            st.pyplot(fig)

        # Matriz de correlación y heatmap
        if st.checkbox("Mostrar matriz de correlación"):
            st.write("Matriz de correlación:")
            corr_matrix = df.corr()
            st.write(corr_matrix)

            st.write("Heatmap de la matriz de correlación:")
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # Autocorrelación (solo si es una serie temporal o numérica)
        if st.checkbox("Mostrar autocorrelación"):
            fig, ax = plt.subplots()
            plot_acf(df[column].dropna(), ax=ax)
            st.pyplot(fig)

        # Gráfico de dispersión interactivo con Plotly
        if st.checkbox("Mostrar gráfico interactivo"):
            x_axis = st.selectbox("Selecciona el eje X", df.columns)
            y_axis = st.selectbox("Selecciona el eje Y", df.columns)

            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Gráfico interactivo de {x_axis} vs {y_axis}")
            st.plotly_chart(fig)

    # Opción 2: Modelado Predictivo
    elif option == "Modelado Predictivo":
        st.write("### Modelado Predictivo")

        # Convertir la columna de fecha a tipo datetime si es necesario
        date_column = st.selectbox("Selecciona la columna de fecha (si aplica)", [None] + df.columns.tolist())
        if date_column is not None:
            df[date_column] = pd.to_datetime(df[date_column])

        # Seleccionar variable objetivo
        target = st.selectbox("Selecciona la variable objetivo (dependiente)", df.columns)

        # Seleccionar las variables predictoras
        features = st.multiselect("Selecciona las variables predictoras (independientes)", df.columns.tolist(), default=df.columns[:-1])

        if len(features) > 0 and target:
            X = df[features]
            y = df[target]

            # Selección de la observación de prueba (se excluye del entrenamiento)
            st.write("## Selección de Observación de Prueba")

            # Si hay una columna de fecha, permitimos la selección por fecha
            if date_column is not None:
                test_date = st.selectbox("Selecciona la fecha de prueba", df[date_column].sort_values())
                test_index = df[date_column].index[df[date_column] == test_date][0]
            else:
                test_index = st.slider("Selecciona el índice de la observación de prueba", min_value=0, max_value=len(df)-1, value=len(df)-2)
                test_date = None

            # Entrenamiento con los datos anteriores a la observación seleccionada
            X_train = X[:test_index]  # Datos de entrenamiento
            y_train = y[:test_index]  # Variable dependiente de entrenamiento
            X_test = X.iloc[test_index]  # Dato de prueba

            # Definir automáticamente 'y_test' si no está presente
            if 'y_test' not in locals():  
                y_test = y.iloc[test_index]  # Definición de y_test

            st.write(f"Entrenando con observaciones anteriores a {test_date or test_index}")

            # Explicación de modelos disponibles
            st.write("## Explicación de Modelos")

            # Expander para modelos de series temporales
            with st.expander("¿Qué es ARIMA?"):
                st.write("ARIMA es un modelo de predicción de series temporales que combina tres componentes: la autoregresión (AR), la diferenciación (I) y el promedio móvil (MA). Es adecuado para datos sin estacionalidad.")

            with st.expander("¿Qué es Exponential Smoothing (ETS)?"):
                st.write("ETS (Suavización Exponencial) se utiliza para capturar tendencias y estacionalidad en series temporales.")

            with st.expander("¿Qué es SARIMA?"):
                st.write("SARIMA es una extensión de ARIMA que incorpora componentes estacionales para series temporales.")

            # Expander para modelos tradicionales
            with st.expander("¿Qué es Regresión Lineal?"):
                st.write("La regresión lineal asume una relación lineal entre la variable dependiente y una o más variables independientes.")

            with st.expander("¿Qué es Árbol de Decisión?"):
                st.write("Un Árbol de Decisión divide los datos en subconjuntos basados en decisiones sobre las variables predictoras.")

            with st.expander("¿Qué es Bosque Aleatorio?"):
                st.write("El Bosque Aleatorio combina varios árboles de decisión para mejorar la precisión y evitar el sobreajuste.")

            with st.expander("¿Qué es K-Nearest Neighbors (KNN)?"):
                st.write("KNN predice basándose en los K vecinos más cercanos a la muestra que se quiere predecir.")

            with st.expander("¿Qué es Regresión Polinómica?"):
                st.write("La regresión polinómica es una extensión de la regresión lineal para modelar relaciones no lineales.")

            # Seleccionar uno o más modelos de predicción
            model_choices = st.multiselect("Selecciona uno o más modelos", 
                                           ["Regresión Lineal", "Árbol de Decisión", "Bosque Aleatorio", 
                                            "K-Nearest Neighbors", "Regresión Polinómica", 
                                            "ARIMA", "Exponential Smoothing (ETS)", "SARIMA"])

            # Inicializar el diccionario de modelos entrenados fuera del condicional
            trained_models_dict = {}

            # Paso del código que permite seleccionar los modelos y entrenar
            if st.button("Entrenar Modelos"):
                # Entrenamiento de los modelos seleccionados
                for model_choice in model_choices:
                    st.write(f"### Entrenando el modelo: {model_choice}")

                    # Modelos tradicionales
                    if model_choice == "Regresión Lineal":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        trained_models_dict[model_choice] = model

                    elif model_choice == "Árbol de Decisión":
                        model = DecisionTreeRegressor()
                        model.fit(X_train, y_train)
                        trained_models_dict[model_choice] = model

                    elif model_choice == "Bosque Aleatorio":
                        model = RandomForestRegressor()
                        model.fit(X_train, y_train)
                        trained_models_dict[model_choice] = model

                    elif model_choice == "K-Nearest Neighbors":
                        model = KNeighborsRegressor(n_neighbors=5)
                        model.fit(X_train, y_train)
                        trained_models_dict[model_choice] = model

                    elif model_choice == "Regresión Polinómica":
                        poly = PolynomialFeatures(degree=2)
                        X_poly_train = poly.fit_transform(X_train)
                        model = LinearRegression()
                        model.fit(X_poly_train, y_train)
                        trained_models_dict[model_choice] = (model, poly)

                    # Modelos de series temporales
                    elif model_choice == "ARIMA":
                        p = st.slider("Selecciona el valor de p (autoregresión)", 0, 5, 1, key="p_arima")
                        d = st.slider("Selecciona el valor de d (diferenciación)", 0, 2, 1, key="d_arima")
                        q = st.slider("Selecciona el valor de q (media móvil)", 0, 5, 1, key="q_arima")
                        model_arima = ARIMA(df[target][:test_index], order=(p, d, q))
                        model_fit = model_arima.fit()
                        trained_models_dict[model_choice] = model_fit

                    elif model_choice == "Exponential Smoothing (ETS)":
                        model_ets = ExponentialSmoothing(df[target][:test_index], trend="add", seasonal="add", seasonal_periods=12)
                        model_fit = model_ets.fit()
                        trained_models_dict[model_choice] = model_fit

                    elif model_choice == "SARIMA":
                        p = st.slider("Selecciona el valor de p (autoregresión)", 0, 5, 1, key="p_sarima")
                        d = st.slider("Selecciona el valor de d (diferenciación)", 0, 2, 1, key="d_sarima")
                        q = st.slider("Selecciona el valor de q (media móvil)", 0, 5, 1, key="q_sarima")
                        P = st.slider("Selecciona el valor de P (estacionalidad)", 0, 5, 1, key="P_sarima")
                        D = st.slider("Selecciona el valor de D (diferenciación estacional)", 0, 2, 1, key="D_sarima")
                        Q = st.slider("Selecciona el valor de Q (media móvil estacional)", 0, 5, 1, key="Q_sarima")
                        m = st.slider("Selecciona el período estacional (m)", 0, 12, 12, key="m_sarima")
                        model_sarima = SARIMAX(df[target][:test_index], order=(p, d, q), seasonal_order=(P, D, Q, m))
                        model_fit = model_sarima.fit()
                        trained_models_dict[model_choice] = model_fit

                st.write("Modelos entrenados correctamente.")

            # Mostrar los resultados de los modelos entrenados
            st.write("### Resultados de los Modelos Seleccionados")

            real_values = [y_test]
            model_predictions = []

            for model_choice, model in trained_models_dict.items():
                if model_choice in ["Regresión Polinómica"]:
                    model, poly = trained_models_dict[model_choice]
                    X_poly_test = poly.transform([X_test])
                    prediction_value = model.predict(X_poly_test)[0]
                elif model_choice in ["ARIMA", "Exponential Smoothing (ETS)", "SARIMA"]:
                    try:
                        if model_choice == "ARIMA" or model_choice == "SARIMA":
                            forecast = model.get_forecast(steps=1)
                            prediction_value = forecast.predicted_mean.iloc[0]
                        else:
                            prediction_value = model.forecast(steps=1)[0]

                    except Exception as e:
                        st.write(f"Error al predecir con {model_choice}: {e}")
                        prediction_value = None

                else:
                    prediction_value = model.predict([X_test])[0]

                y_test_value = y_test if not isinstance(y_test, pd.Series) else y_test.item()

                mae = mean_absolute_error([y_test_value], [prediction_value])
                mse = mean_squared_error([y_test_value], [prediction_value])

                diff_percent = abs((y_test_value - prediction_value) / y_test_value) * 100

                st.write(f"**{model_choice}:**")
                st.write(f"Valor Real: {y_test_value}")
                st.write(f"Predicción: {prediction_value}")
                st.write(f"Error Absoluto Medio (MAE): {mae:.4f}")
                st.write(f"Error Cuadrático Medio (MSE): {mse:.4f}")
                st.write(f"Diferencia porcentual: {diff_percent:.2f}%")
                st.write("---")

                model_predictions.append(prediction_value)

            # Gráfico comparativo
            st.write("### Comparación Gráfica de las Predicciones con el Valor Real")
            df_graph = pd.DataFrame({
                'Modelo': list(trained_models_dict.keys()),
                'Predicciones': model_predictions,
                'Valor Real': real_values * len(trained_models_dict)
            })

            st.bar_chart(df_graph.set_index('Modelo'))

            # Gráfico de dispersión interactivo usando Plotly
            fig = px.scatter(df_graph, x='Modelo', y='Predicciones', labels={'Predicciones': 'Predicción'}, title="Predicciones de cada Modelo vs Valor Real")
            fig.add_scatter(x=df_graph['Modelo'], y=df_graph['Valor Real'], mode='lines+markers', name='Valor Real')
            st.plotly_chart(fig)

            # Gráfico de líneas usando Matplotlib para análisis más visual
            st.write("### Gráfico de Líneas - Comparación de Predicciones")
            plt.figure(figsize=(10, 5))
            plt.plot(df_graph['Modelo'], df_graph['Predicciones'], label='Predicciones', marker='o')
            plt.axhline(y=y_test, color='r', linestyle='--', label='Valor Real')
            plt.title('Comparación de Predicciones por Modelo')
            plt.xlabel('Modelo')
            plt.ylabel('Valor Predicho')
            plt.legend()
            st.pyplot(plt)

# Predicción de un valor no conocido
st.write("### Predicción de un valor no conocido")

# Asegurarse de que el diccionario `trained_models_dict` esté definido correctamente
if 'trained_models_dict' not in locals():
    trained_models_dict = {}

# Omitimos el uso del selectbox si los modelos ya han sido entrenados
st.write("Seleccionando automáticamente los modelos previamente entrenados")

# Verificar si es una serie temporal
if date_column is not None:
    st.write("Esta es una serie temporal. Introduce una fecha futura para hacer una predicción.")
    future_date = st.date_input("Selecciona la fecha que deseas predecir")

    if st.button("Predecir con fecha futura"):
        st.write(f"Prediciendo el valor para la fecha: {future_date}")

        for model_choice, model in trained_models_dict.items():
            st.write(f"### Predicción con {model_choice}")

            if model_choice == "ARIMA":
                prediction = model.get_forecast(steps=1).predicted_mean[0]
            elif model_choice == "Exponential Smoothing (ETS)":
                prediction = model.forecast(steps=1)[0]
            elif model_choice == "SARIMA":
                prediction = model.get_forecast(steps=1).predicted_mean[0]

            st.write(f"Predicción del modelo {model_choice} para la fecha {future_date}: {prediction}")

else:
    st.write("Introduce los valores para las variables predictoras del nuevo dato")

    # Crear inputs dinámicos para cada una de las variables predictoras
    new_data = []
    for feature in features:
        value = st.number_input(f"Introduce un valor para {feature}")
        new_data.append(value)

    # Botón para entrenar los modelos y predecir
    if st.button("Entrenar y Predecir con nuevo dato"):
        new_data = np.array(new_data).reshape(1, -1)

        # Entrenar de nuevo los modelos con el valor ingresado
        trained_models_dict = {}

        # Entrenamiento de los modelos seleccionados
        for model_choice in model_choices:
            st.write(f"### Entrenando y prediciendo con el modelo: {model_choice}")

            # Modelos tradicionales
            if model_choice == "Regresión Lineal":
                model = LinearRegression()
                model.fit(X, y)  # Entrenamos nuevamente
                trained_models_dict[model_choice] = model
                prediction = model.predict(new_data)[0]
                st.write(f"Predicción del modelo {model_choice} para el nuevo dato: {prediction}")

            elif model_choice == "Árbol de Decisión":
                model = DecisionTreeRegressor()
                model.fit(X, y)  # Entrenamos nuevamente
                trained_models_dict[model_choice] = model
                prediction = model.predict(new_data)[0]
                st.write(f"Predicción del modelo {model_choice} para el nuevo dato: {prediction}")

            elif model_choice == "Bosque Aleatorio":
                model = RandomForestRegressor()
                model.fit(X, y)  # Entrenamos nuevamente
                trained_models_dict[model_choice] = model
                prediction = model.predict(new_data)[0]
                st.write(f"Predicción del modelo {model_choice} para el nuevo dato: {prediction}")

            elif model_choice == "K-Nearest Neighbors":
                model = KNeighborsRegressor(n_neighbors=5)
                model.fit(X, y)  # Entrenamos nuevamente
                trained_models_dict[model_choice] = model
                prediction = model.predict(new_data)[0]
                st.write(f"Predicción del modelo {model_choice} para el nuevo dato: {prediction}")

            elif model_choice == "Regresión Polinómica":
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)  # Entrenamos nuevamente
                trained_models_dict[model_choice] = (model, poly)
                new_data_poly = poly.transform(new_data)
                prediction = model.predict(new_data_poly)[0]
                st.write(f"Predicción del modelo {model_choice} para el nuevo dato: {prediction}")
