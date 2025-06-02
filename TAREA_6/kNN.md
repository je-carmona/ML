**TAREA 6: kNN**

_Enlace Google Colab:_ https://colab.research.google.com/drive/13qG3ZuJTJoWz-HKcq-l0y6kX31Pt3sDx?usp=sharing

**6.1.** How can you evaluate the hypothesis that the problem with almost all predictions giving `Medium` is due to the disproportionate data in that column?

Se hace un análisis del balance de datos, si estos resultan tener una alter dispersion puede establecerse que debido a esto los rsultados de la prediccion predominan en Medium, si los datos no son dispersos quiere decir que es necesario realizar ajustes a los parametros de entrenamiento del modelo. 

            import polars as pl
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Configuración de visualización
            plt.style.use('ggplot')  # Usamos un estilo más genérico que esté disponible
            sns.set_theme(style="whitegrid")
            sns.set_palette("husl")
            
            # Cargar el dataset
            try:
                df = pl.read_csv("Students_Grading_Dataset.csv")
            except FileNotFoundError:
                print("Error: No se encontró el archivo 'Students_Grading_Dataset.csv'")
                print("Asegúrate de que el archivo esté en el mismo directorio que este script.")
                exit()
            
            ## Análisis de Distribución de Variables Numéricas
            
            # 1. Estadísticas descriptivas básicas
            print("\nEstadísticas descriptivas para variables numéricas:")
            numeric_stats = df.select(pl.col(pl.Float64, pl.Int64)).describe()
            print(numeric_stats)
            
            # 2. Visualización de distribuciones
            def plot_numeric_distribution(column_name):
                plt.figure(figsize=(10, 6))
                sns.histplot(df[column_name].to_numpy(), kde=True, bins=30)
                plt.title(f'Distribución de {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Frecuencia')
                plt.tight_layout()
                plt.show()
            
            numeric_columns = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 
                               'Assignments_Avg', 'Quizzes_Avg', 'Total_Score', 
                               'Study_Hours_per_Week', 'Stress_Level (1-10)', 
                               'Sleep_Hours_per_Night']
            
            for col in numeric_columns:
                if col in df.columns:
                    plot_numeric_distribution(col)
                else:
                    print(f"Advertencia: La columna {col} no existe en el dataset")
            
            ## Análisis de Distribución de Variables Categóricas
            
            # 1. Conteo de frecuencias para variables categóricas
            print("\nDistribución de variables categóricas:")
            categorical_columns = ['Gender', 'Department', 'Grade', 'Extracurricular_Activities',
                                  'Internet_Access_at_Home', 'Parent_Education_Level',
                                  'Family_Income_Level']
            
            for col in categorical_columns:
                if col in df.columns:
                    print(f"\nDistribución de {col}:")
                    print(df[col].value_counts().sort(col))
                else:
                    print(f"Advertencia: La columna {col} no existe en el dataset")
            
            # 2. Visualización de distribuciones categóricas
            def plot_categorical_distribution(column_name):
                plt.figure(figsize=(10, 6))
                # Convertir a DataFrame de Pandas y obtener conteos
                value_counts = df[column_name].value_counts().sort(column_name).to_pandas()
                # Renombrar columnas para claridad
                value_counts.columns = ['category', 'count']
                
                sns.barplot(x='category', y='count', data=value_counts)
                plt.title(f'Distribución de {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Conteo')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            
            for col in categorical_columns:
                if col in df.columns:
                    plot_categorical_distribution(col)
                else:
                    print(f"Advertencia: La columna {col} no existe en el dataset")
            
            ## Análisis de correlaciones entre variables numéricas
            print("\nMatriz de correlación:")
            numeric_df = df.select(pl.col(pl.Float64, pl.Int64))
            correlation_matrix = numeric_df.corr()
            print(correlation_matrix)
            
            # Visualización de la matriz de correlación
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix.to_pandas(), annot=True, fmt=".2f", 
                        cmap='coolwarm', center=0, vmin=-1, vmax=1)
            plt.title('Matriz de Correlación')
            plt.tight_layout()
            plt.show()
            
            ## Análisis por departamento
            print("\nEstadísticas por departamento:")
            if 'Department' in df.columns:
                department_stats = df.group_by('Department').agg([
                    pl.col('Total_Score').mean().alias('Promedio_Total_Score'),
                    pl.col('Total_Score').median().alias('Mediana_Total_Score'),
                    pl.col('Grade').value_counts().alias('Distribucion_Grados')
                ])
                print(department_stats)
            else:
                print("Advertencia: La columna 'Department' no existe en el dataset")
            
            ## Análisis de distribución de notas (Grade) por género
            print("\nDistribución de notas por género:")
            if 'Gender' in df.columns and 'Grade' in df.columns:
                # Convertir a Pandas para facilitar la visualización
                df_pd = df.select(['Gender', 'Grade']).to_pandas()
                
                plt.figure(figsize=(10, 6))
                sns.countplot(data=df_pd, x='Grade', hue='Gender')
                plt.title('Distribución de Notas por Género')
                plt.tight_layout()
                plt.show()
            else:
                print("Advertencia: Las columnas 'Gender' o 'Grade' no existen en el dataset")

                print("")
                print("Desarrollado por: J.E. Carmona-Álvarez")
**Resultados:**

![image](https://github.com/user-attachments/assets/b04405be-ad56-4457-93e7-95b982408e63)

![image](https://github.com/user-attachments/assets/29163637-cafa-42a0-84d3-daa064f6c19a)

![image](https://github.com/user-attachments/assets/d1524222-5829-4c48-b760-2190f146ae68)

![image](https://github.com/user-attachments/assets/3dd582c1-bf00-4f86-ad0e-a53f4d9f3b0b)

![image](https://github.com/user-attachments/assets/834ca813-10f2-481a-9c00-8cc87e6fa816)

![image](https://github.com/user-attachments/assets/34664e87-22ff-4ee1-a7dc-c4d8e50edce7)

![image](https://github.com/user-attachments/assets/516dabdb-bc33-4427-9c71-957d1430f673)

![image](https://github.com/user-attachments/assets/3f7bd53d-56e4-4a8e-8ff7-8ed52ef61c60)

![image](https://github.com/user-attachments/assets/34edab3c-c051-4801-b044-86c0bf7121ee)

![image](https://github.com/user-attachments/assets/1e4de661-881e-4afb-89aa-d7eef328b29e)


![image](https://github.com/user-attachments/assets/6382349a-269e-4c6a-826b-ecfd485cac96)

![image](https://github.com/user-attachments/assets/be2a3997-54d6-4369-868e-73c12392bec9)

Desarrollado por: J.E. Carmona-Álvarez

**Análisis de la hipótesis:** 

Una vez obtenidos los resultados del análisis de desbalance en los datos, se puede observar que la categoría 'Medium' en la variable 'Family_Income_Level' es la más frecuente en el conjunto de datos original. Este desequilibrio introduce un sesgo en el modelo, que tiende a predecir con mayor frecuencia dicha categoría. Esto se refleja en la matriz de confusión, donde la mayoría de las predicciones incorrectas pertenecen a la clase 'Medium'.

La alta tasa de error del modelo entrenado sugiere que este sesgo está generando falsos positivos, en los que se predice 'Medium' cuando en realidad corresponde a otra categoría. Esta situación persiste incluso al variar el valor de **k**, lo cual indica que el modelo sigue favoreciendo la categoría predominante. Por lo tanto, la configuración actual del modelo no es satisfactoria mientras no se aborde el problema de desbalance de clases.

Por tanto, se puede afirmar que la hipótesis de que la mayoría de las predicciones corresponden a la categoría 'Medium' se debe al desbalance en los datos. El modelo está aprendiendo a priorizar la clase mayoritaria para minimizar el error global, aunque esto implique un rendimiento deficiente en la clasificación de las clases minoritarias.

**6.2.** Predict the numeric variable 'Study_Hours_per_Week_normalized', determine the error for each **k** and choice the best **k**.

Para predecir la variable numérica Study_Hours_per_Week_normalized y determinar el mejor valor de k, es necesario: 

**1.** Preparar los datos

            import polars as pl
            import numpy as np
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            import matplotlib.pyplot as plt
            
            # Seleccionar características y variable objetivo
            features = df5.select([
                "Stress_Level (1-10)_normalized",
                "Grade_A", "Grade_B", "Grade_C", "Grade_D", "Grade_F",
                "Income_Low", "Income_Medium", "Income_High"
            ])
            
            target = df5.select("Study_Hours_per_Week_normalized")
            
            # Convertir a pandas para compatibilidad con sklearn
            X = features.to_pandas()
            y = target.to_pandas().values.ravel()  # Convertir a array 1D
            
            # Dividir en conjuntos de entrenamiento y prueba (80-20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**2.** Evaluación de Diferentes Valores de k

            k_values = range(1, 31)  # Probaremos k de 1 a 30
            mse_values = []
            
            for k in k_values:
                # Crear y entrenar modelo KNN
                knn = KNeighborsRegressor(n_neighbors=k)
                knn.fit(X_train, y_train)
                
                # Predecir y calcular error cuadrático medio
                y_pred = knn.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_values.append(mse)
                
                print(f"k = {k:2d} - MSE: {mse:.6f}")

**Resultados:**

k =  1 - MSE: 0.158812
k =  2 - MSE: 0.117504
k =  3 - MSE: 0.107019
k =  4 - MSE: 0.101223
k =  5 - MSE: 0.096867
k =  6 - MSE: 0.093472
k =  7 - MSE: 0.092825
k =  8 - MSE: 0.091702
k =  9 - MSE: 0.090834
k = 10 - MSE: 0.088583
k = 11 - MSE: 0.088567
k = 12 - MSE: 0.088815
k = 13 - MSE: 0.087819
k = 14 - MSE: 0.087300
k = 15 - MSE: 0.087611
k = 16 - MSE: 0.087145
k = 17 - MSE: 0.086884
k = 18 - MSE: 0.086991
k = 19 - MSE: 0.087020
k = 20 - MSE: 0.086347
k = 21 - MSE: 0.086309
k = 22 - MSE: 0.086260
k = 23 - MSE: 0.086264
k = 24 - MSE: 0.086463
k = 25 - MSE: 0.086841
k = 26 - MSE: 0.086853
k = 27 - MSE: 0.086623
k = 28 - MSE: 0.086541
k = 29 - MSE: 0.086243
k = 30 - MSE: 0.086466

**3.** Selección del Mejor k

            # Encontrar el k con menor MSE
            best_k = k_values[np.argmin(mse_values)]
            best_mse = min(mse_values)
            
            print(f"\nEl mejor valor de k es {best_k} con un MSE de {best_mse:.6f}")
            
            # Graficar MSE vs k
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, mse_values, marker='o')
            plt.axvline(x=best_k, color='r', linestyle='--', label=f'Mejor k = {best_k}')
            plt.xlabel('Valor de k')
            plt.ylabel('Error Cuadrático Medio (MSE)')
            plt.title('Error vs Valor de k en KNN Regresión')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            print("")
            print("Desarrollado por: J.E. Carmona-Álvarez")

El mejor valor de k es 29 con un MSE de 0.086243

![image](https://github.com/user-attachments/assets/7443fead-503c-4090-88e0-c598b4ecab68)

Desarrollado por: J.E. Carmona-Álvarez

**6.3.** Add new variables that can improve performance.

Ajuste del entrenamiento del Modelo usando las librerias de Polars 

            #Desarrollado con la libreria de POLARS
            
            #Librerias usadas
            
            import polars as pl
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import GridSearchCV
            from scipy.spatial.distance import euclidean, hamming
            import time
            
            # Configuración inicial
            plt.style.use('ggplot')
            plt.rcParams['figure.figsize'] = (12, 6)
            np.random.seed(42)
            
            ## 1. Carga y exploración inicial de datos
            def load_and_explore_data(file_path):
                # Cargar datos
                df = pl.read_csv(file_path)
                
                # Exploración inicial
                print("=== Información del Dataset ===")
                print(f"Filas: {df.height}, Columnas: {df.width}")
                print("\n=== Primeras filas ===")
                print(df.head())
                print("\n=== Estadísticas descriptivas ===")
                print(df.describe())
                print("\n=== Tipos de datos ===")
                print(df.schema)
                print("\n=== Valores faltantes ===")
                print(df.null_count())
                
                return df
            
            # Cargar y explorar datos
            df = load_and_explore_data("Students_Grading_Dataset.csv")
            
            ## 2. Selección y análisis de columnas relevantes
            def select_relevant_columns(df):
                # Columnas a mantener
                relevant_cols = [
                    'Student_ID', 'Gender', 'Age', 'Department', 'Attendance (%)',
                    'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg',
                    'Participation_Score', 'Projects_Score', 'Total_Score', 'Grade',
                    'Study_Hours_per_Week', 'Extracurricular_Activities', 
                    'Internet_Access_at_Home', 'Parent_Education_Level',
                    'Family_Income_Level', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night'
                ]
                
                # Filtrar columnas
                df = df.select(relevant_cols)
                
                # Verificar
                print("\n=== Columnas seleccionadas ===")
                print(df.columns)
                
                return df
            
            df = select_relevant_columns(df)
            
            ## 3. Análisis estadístico y visualización
            def analyze_and_visualize(df):
                # Convertir a pandas para visualización
                pdf = df.to_pandas()
                
                # Distribución de notas
                plt.figure(figsize=(10, 6))
                sns.countplot(data=pdf, x='Grade', order=sorted(pdf['Grade'].unique()))
                plt.title('Distribución de Calificaciones')
                plt.show()
                
                # Relación entre asistencia y nota final
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=pdf, x='Grade', y='Attendance (%)', order=sorted(pdf['Grade'].unique()))
                plt.title('Asistencia por Calificación')
                plt.show()
                
                # Correlación entre variables numéricas
                numeric_cols = pdf.select_dtypes(include=['float64', 'int64']).columns
                plt.figure(figsize=(12, 8))
                sns.heatmap(pdf[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
                plt.title('Matriz de Correlación')
                plt.show()
                
                # Distribución por departamento
                plt.figure(figsize=(12, 6))
                sns.countplot(data=pdf, x='Department', hue='Grade')
                plt.title('Distribución de Calificaciones por Departamento')
                plt.xticks(rotation=45)
                plt.show()
            
            analyze_and_visualize(df)
            
            ## 4. Transformación definitiva de datos
            def transform_data(df):
                # Convertir a pandas temporalmente para transformaciones
                pdf = df.to_pandas()
                
                # Codificación de variables categóricas con LabelEncoder
                categorical_cols = ['Gender', 'Department', 'Extracurricular_Activities', 
                                  'Internet_Access_at_Home', 'Parent_Education_Level', 
                                  'Family_Income_Level']
                
                label_encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    pdf[col] = le.fit_transform(pdf[col])
                    label_encoders[col] = le
                
                # Convertir Grade a numérico (A=4, B=3, C=2, D=1, F=0)
                grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
                pdf['Grade'] = pdf['Grade'].map(grade_map)
                
                # Volver a Polars
                df = pl.from_pandas(pdf)
                
                # Verificar transformación
                print("\n=== Datos después de transformación ===")
                print(df.head())
                
                return df, label_encoders, grade_map
            
            df, label_encoders, grade_map = transform_data(df)
            
            ## 5. Preparación para modelado
            def prepare_for_modeling(df):
                # Convertir a pandas para usar con scikit-learn
                pdf = df.to_pandas()
                
                # Definir características (X) y objetivo (y)
                features = ['Gender', 'Age', 'Department', 'Attendance (%)', 'Midterm_Score', 
                           'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score',
                           'Projects_Score', 'Study_Hours_per_Week', 'Extracurricular_Activities',
                           'Internet_Access_at_Home', 'Parent_Education_Level', 
                           'Family_Income_Level', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
                
                target = 'Grade'
                
                X = pdf[features]
                y = pdf[target]
                
                # Dividir en conjuntos de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Escalar características
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                return X_train, X_test, y_train, y_test, scaler, features, target
            
            X_train, X_test, y_train, y_test, scaler, features, target = prepare_for_modeling(df)
            
            ## 6. Función de distancia - Combinación de Euclidiana y Hamming
            def combined_distance(x, y, alpha=0.5):
                """
                Combina distancia Euclidiana y Hamming con un factor de peso alpha.
                alpha: peso para la distancia Euclidiana (1-alpha para Hamming)
                """
                eucl_dist = euclidean(x, y)
                ham_dist = hamming(x, y) * len(x)  # hamming devuelve promedio, multiplicamos por longitud
                
                return alpha * eucl_dist + (1 - alpha) * ham_dist
            
            ## 7. Implementación de KNN con distancia personalizada
            class CustomKNN:
                def __init__(self, k=5, alpha=0.5):
                    self.k = k
                    self.alpha = alpha  # Peso para distancia combinada
                    
                def fit(self, X, y):
                    self.X_train = X
                    self.y_train = y
                    
                def predict(self, X):
                    predictions = []
                    for x in X:
                        # Calcular distancias a todos los puntos de entrenamiento
                        distances = [combined_distance(x, x_train, self.alpha) 
                                    for x_train in self.X_train]
                        
                        # Obtener los k índices más cercanos
                        k_indices = np.argsort(distances)[:self.k]
                        
                        # Obtener las etiquetas de los k vecinos más cercanos
                        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
                        
                        # Votación mayoritaria
                        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
                        predictions.append(most_common)
                        
                    return np.array(predictions)
            
            ## 8. Entrenamiento y evaluación del modelo
            def train_and_evaluate(X_train, X_test, y_train, y_test, k=5, alpha=0.5):
                # Modelo KNN estándar - Euclidiana
                print("\n=== KNN estándar (Euclidiana) ===")
                start_time = time.time()
                knn_std = KNeighborsClassifier(n_neighbors=k)
                knn_std.fit(X_train, y_train)
                y_pred_std = knn_std.predict(X_test)
                
                print(f"Tiempo de entrenamiento: {time.time() - start_time:.2f} segundos")
                print("Exactitud:", accuracy_score(y_test, y_pred_std))
                print("\nReporte de clasificación:")
                print(classification_report(y_test, y_pred_std))
                
                # Modelo KNN personalizado - Combinación Euclidiana + Hamming 
                print("\n=== KNN personalizado (Combinación Euclidiana + Hamming) ===")
                start_time = time.time()
                knn_custom = CustomKNN(k=k, alpha=alpha)
                knn_custom.fit(X_train, y_train)
                y_pred_custom = knn_custom.predict(X_test)
                
                print(f"Tiempo de entrenamiento: {time.time() - start_time:.2f} segundos")
                print("Exactitud:", accuracy_score(y_test, y_pred_custom))
                print("\nReporte de clasificación:")
                print(classification_report(y_test, y_pred_custom))
                
                # Matriz de confusión para el modelo personalizado
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred_custom)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=grade_map.keys(), yticklabels=grade_map.keys())
                plt.title('Matriz de Confusión (KNN Personalizado)')
                plt.xlabel('Predicho')
                plt.ylabel('Real')
                plt.show()
                
                return knn_std, knn_custom, y_pred_std, y_pred_custom
            
            knn_std, knn_custom, y_pred_std, y_pred_custom = train_and_evaluate(
                X_train, X_test, y_train, y_test, k=5, alpha=0.6
            )
            
            ## 9. Optimización del modelo para la búsqueda del mejor k y alpha
            def optimize_model(X_train, y_train):
                # Definir parámetros a probar
                param_grid = {
                    'n_neighbors': range(3, 15, 2),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
                
                # GridSearchCV para encontrar los mejores parámetros
                grid = GridSearchCV(
                    KNeighborsClassifier(), 
                    param_grid, 
                    cv=5, 
                    scoring='accuracy', 
                    n_jobs=-1
                )
                
                grid.fit(X_train, y_train)
                
                print("\n Mejores parámetros para KNN estándar")
                print(grid.best_params_)
                print("Mejor exactitud:", grid.best_score_)
                
                # Evaluar con los mejores parámetros
                best_knn = grid.best_estimator_
                
                return best_knn
            
            best_knn = optimize_model(X_train, y_train)
            
            # Evaluar el mejor modelo en el conjunto de prueba
            y_pred_best = best_knn.predict(X_test)
            print("\nRendimiento del mejor modelo en conjunto de prueba")
            print("Exactitud:", accuracy_score(y_test, y_pred_best))
            print("\nReporte de clasificación:")
            print(classification_report(y_test, y_pred_best))
            
            ## 10. Visualización de la búsqueda del mejor k
            def plot_k_search(X_train, y_train, X_test, y_test):
                # Probar diferentes valores de k
                k_values = range(1, 30)
                train_scores = []
                test_scores = []
                
                for k in k_values:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train, y_train)
                    train_scores.append(knn.score(X_train, y_train))
                    test_scores.append(knn.score(X_test, y_test))
                
                # Graficar resultados
                plt.figure(figsize=(12, 6))
                plt.plot(k_values, train_scores, label='Entrenamiento', marker='o')
                plt.plot(k_values, test_scores, label='Prueba', marker='o')
                plt.xlabel('Valor de k')
                plt.ylabel('Exactitud')
                plt.title('Búsqueda del mejor k para KNN')
                plt.legend()
                plt.grid(True)
                plt.show()
                
                # Encontrar el k óptimo
                optimal_k = k_values[np.argmax(test_scores)]
                print(f"\nEl valor óptimo de k es: {optimal_k}")
                
                return optimal_k
            
            optimal_k = plot_k_search(X_train, y_train, X_test, y_test)
            
            ## 11. Evaluación final con el mejor modelo
            print("\n Evaluación final con el mejor modelo")
            final_knn = KNeighborsClassifier(
                n_neighbors=optimal_k,
                weights='distance',
                metric='euclidean'
            )
            final_knn.fit(X_train, y_train)
            y_pred_final = final_knn.predict(X_test)
            
            print("Exactitud:", accuracy_score(y_test, y_pred_final))
            print("\nReporte de clasificación:")
            print(classification_report(y_test, y_pred_final))
            
            # Matriz de confusión final
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred_final)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=grade_map.keys(), yticklabels=grade_map.keys())
            plt.title('Matriz de Confusión para el Modelo Final')
            plt.xlabel('Predicho')
            plt.ylabel('Real')
            plt.show()
            
            print("")
            print("Desarrollado por: J.E. Carmona-Álvarez")

**Resultados:**

=== Información del Dataset ===
Filas: 5000, Columnas: 23

=== Primeras filas ===
shape: (5, 23)

![image](https://github.com/user-attachments/assets/87e8a09d-5b02-40a8-a42c-96075abbba83)

=== Estadísticas descriptivas ===
shape: (9, 24)

![image](https://github.com/user-attachments/assets/bcc63ddd-560a-40ca-bbb7-bcbd76afd95a)

=== Tipos de datos ===
Schema([('Student_ID', String), ('First_Name', String), ('Last_Name', String), ('Email', String), ('Gender', String), ('Age', Int64), ('Department', String), ('Attendance (%)', Float64), ('Midterm_Score', Float64), ('Final_Score', Float64), ('Assignments_Avg', Float64), ('Quizzes_Avg', Float64), ('Participation_Score', Float64), ('Projects_Score', Float64), ('Total_Score', Float64), ('Grade', String), ('Study_Hours_per_Week', Float64), ('Extracurricular_Activities', String), ('Internet_Access_at_Home', String), ('Parent_Education_Level', String), ('Family_Income_Level', String), ('Stress_Level (1-10)', Int64), ('Sleep_Hours_per_Night', Float64)])

=== Valores faltantes ===
shape: (1, 23)

![image](https://github.com/user-attachments/assets/bfd11cbd-84d3-41dc-9c37-492a85383b73)

=== Columnas seleccionadas ===
['Student_ID', 'Gender', 'Age', 'Department', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score', 'Grade', 'Study_Hours_per_Week', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']

![image](https://github.com/user-attachments/assets/1a0e2820-cfaa-4429-b413-855a4dca848c)
![image](https://github.com/user-attachments/assets/ffdd1154-c1a2-4c2d-ae20-a5abce94de81)
![image](https://github.com/user-attachments/assets/64d9ad19-564b-4157-8f8d-09f19a9fc51c)
![image](https://github.com/user-attachments/assets/a1b35f53-9ef4-48b8-b5de-ebc624fe0dab)

=== Datos después de transformación ===
shape: (5, 20)

![image](https://github.com/user-attachments/assets/16e8e00c-1b96-430b-ba6b-675918819bc0)

=== KNN estándar (Euclidiana) ===
Tiempo de entrenamiento: 0.10 segundos
Exactitud: 0.21133333333333335

Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.19      0.27      0.22       301
           1       0.20      0.19      0.20       303
           2       0.18      0.19      0.18       293
           3       0.25      0.21      0.23       304
           4       0.27      0.20      0.23       299

    accuracy                           0.21      1500
   macro avg       0.22      0.21      0.21      1500
weighted avg       0.22      0.21      0.21      1500


=== KNN personalizado (Combinación Euclidiana + Hamming) ===
Tiempo de entrenamiento: 101.63 segundos
Exactitud: 0.22


Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.22      0.33      0.26       301
           1       0.25      0.26      0.26       303
           2       0.20      0.19      0.20       293
           3       0.18      0.14      0.16       304
           4       0.26      0.17      0.20       299

    accuracy                           0.22      1500
   macro avg       0.22      0.22      0.22      1500
weighted avg       0.22      0.22      0.22      1500

![image](https://github.com/user-attachments/assets/2099e4ab-7e49-4a77-b2f1-05e6b04c9f34)

 Mejores parámetros para KNN estándar
{'metric': 'manhattan', 'n_neighbors': 13, 'weights': 'distance'}
Mejor exactitud: 0.21085714285714285

Rendimiento del mejor modelo en conjunto de prueba
Exactitud: 0.22533333333333333

Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.20      0.21      0.21       301
           1       0.27      0.25      0.26       303
           2       0.21      0.20      0.21       293
           3       0.20      0.20      0.20       304
           4       0.25      0.26      0.25       299

    accuracy                           0.23      1500
   macro avg       0.23      0.23      0.23      1500
weighted avg       0.23      0.23      0.23      1500


![image](https://github.com/user-attachments/assets/18a7a370-fd23-4aaa-88fe-4f520fefcfdb)

El valor óptimo de k es: 1

 Evaluación final con el mejor modelo
Exactitud: 0.21866666666666668

Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.21      0.21      0.21       301
           1       0.18      0.16      0.17       303
           2       0.20      0.21      0.20       293
           3       0.27      0.26      0.26       304
           4       0.23      0.26      0.25       299

    accuracy                           0.22      1500
   macro avg       0.22      0.22      0.22      1500
weighted avg       0.22      0.22      0.22      1500

![image](https://github.com/user-attachments/assets/aaf952c0-bb9f-4abd-aeec-38df6d939777)

Desarrollado por: J.E. Carmona-Álvarez

**2.** Ajuste del entrenamiento del Modelo usando las librerias de Polars 

            # Usando Librerias de PANDAS
            
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            from scipy.spatial.distance import mahalanobis
            from scipy.stats import zscore
            
            # Configuración de visualización
            plt.style.use('ggplot')
            plt.rcParams['figure.figsize'] = (12, 6)
            pd.set_option('display.max_columns', 50)
            
            # 1. Carga y exploración inicial de datos
            df = pd.read_csv('Students_Grading_Dataset.csv')
            
            # Mostrar información básica del dataset
            print("="*80)
            print("Información básica del dataset:")
            print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
            print("\nPrimeras 5 filas:")
            print(df.head())
            print("\nResumen estadístico:")
            print(df.describe())
            print("\nTipos de datos y valores nulos:")
            print(df.info())
            
            # 2. Selección y análisis de columnas relevantes
            # Eliminar columnas no relevantes para el modelado
            columns_to_drop = ['Student_ID', 'First_Name', 'Last_Name', 'Email']
            df = df.drop(columns=columns_to_drop)
            
            # Analizar correlación con la variable objetivo (Grade)
            print("\n" + "="*80)
            print("Análisis de correlación con la variable objetivo (Grade):")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            correlation_with_grade = df[numeric_cols].corrwith(df['Total_Score']).sort_values(ascending=False)
            print(correlation_with_grade)
            
            # 3. Análisis estadístico y visualización
            # Visualización de distribución de la variable objetivo
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='Grade', order=sorted(df['Grade'].unique()))
            plt.title('Distribución de Calificaciones (Grade)')
            plt.show()
            
            # Visualización de relaciones entre variables
            plt.figure(figsize=(12, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Matriz de Correlación')
            plt.show()
            
            # Visualización de variables importantes vs Grade
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            sns.boxplot(data=df, x='Grade', y='Total_Score', ax=axes[0, 0])
            sns.boxplot(data=df, x='Grade', y='Midterm_Score', ax=axes[0, 1])
            sns.boxplot(data=df, x='Grade', y='Final_Score', ax=axes[1, 0])
            sns.boxplot(data=df, x='Grade', y='Assignments_Avg', ax=axes[1, 1])
            plt.tight_layout()
            plt.show()
            
            # 4. Transformación de datos
            # Codificación de variables categóricas
            categorical_cols = df.select_dtypes(include=['object']).columns
            print("\n" + "="*80)
            print("Variables categóricas a codificar:", categorical_cols)
            
            label_encoders = {}
            for col in categorical_cols:
                if col != 'Grade':  # No codificamos la variable objetivo todavía
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
            
            # Codificación de la variable objetivo (Grade)
            grade_order = sorted(df['Grade'].unique())
            grade_mapping = {grade: i for i, grade in enumerate(grade_order)}
            df['Grade_encoded'] = df['Grade'].map(grade_mapping)
            
            # Manejo de valores atípicos usando z-score
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            numeric_cols = [col for col in numeric_cols if col != 'Grade_encoded']
            
            for col in numeric_cols:
                z_scores = zscore(df[col])
                df = df[(np.abs(z_scores) < 3)]  # Eliminar outliers con z-score > 3
            
            # 5. Preparación para modelado
            # Separar características y variable objetivo
            X = df.drop(['Grade', 'Grade_encoded'], axis=1)
            y = df['Grade_encoded']
            
            # Dividir en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Escalar características
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 6. Entrenamiento de modelo KNN con distancia personalizada
            # Función de distancia personalizada (Mahalanobis)
            def mahalanobis_distance(x, y, cov_inv):
                diff = x - y
                return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
            
            # Calcular matriz de covarianza inversa para Mahalanobis
            cov_matrix = np.cov(X_train_scaled, rowvar=False)
            cov_inv = np.linalg.inv(cov_matrix)
            
            # Crear función de distancia personalizada para KNN
            def custom_distance(x, y):
                return mahalanobis_distance(x, y, cov_inv)
            
            # Entrenar modelo KNN con distancia personalizada
            knn_custom = KNeighborsClassifier(n_neighbors=5, metric=custom_distance)
            knn_custom.fit(X_train_scaled, y_train)
            
            # 7. Evaluación del modelo
            y_pred = knn_custom.predict(X_test_scaled)
            
            print("\n" + "="*80)
            print("Evaluación del Modelo KNN con Distancia Personalizada:")
            print("\nReporte de Clasificación:")
            print(classification_report(y_test, y_pred, target_names=grade_order))
            
            print("\nMatriz de Confusión:")
            conf_matrix = confusion_matrix(y_test, y_pred)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=grade_order, yticklabels=grade_order)
            plt.title('Matriz de Confusión')
            plt.xlabel('Predicho')
            plt.ylabel('Real')
            plt.show()
            
            print(f"\nPrecisión del modelo: {accuracy_score(y_test, y_pred):.2f}")
            
            # 8. Optimización del modelo
            # Búsqueda de hiperparámetros óptimos
            param_grid = {
                'n_neighbors': range(3, 15),
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # Para distancia Manhattan (1) y Euclidiana (2)
            }
            
            # Usar distancia estándar para la optimización (más rápida)
            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            print("\n" + "="*80)
            print("Optimización de Hiperparámetros:")
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor precisión en validación cruzada: {grid_search.best_score_:.2f}")
            
            # Evaluar modelo optimizado
            best_knn = grid_search.best_estimator_
            y_pred_optimized = best_knn.predict(X_test_scaled)
            
            print("\nEvaluación del Modelo Optimizado:")
            print(classification_report(y_test, y_pred_optimized, target_names=grade_order))
            print(f"Precisión del modelo optimizado: {accuracy_score(y_test, y_pred_optimized):.2f}")
            
            # Visualizar importancia de características (basado en distancia)
            feature_importance = np.mean(np.abs(scaler.inverse_transform(X_train_scaled)), axis=0)
            sorted_idx = np.argsort(feature_importance)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
            plt.title('Importancia de Características (basado en magnitud promedio)')
            plt.xlabel('Magnitud Promedio')
            plt.tight_layout()
            plt.show()
            
            print("")
            print("Desarrollado por: J.E. Carmona-Álvarez")

**Resultados:**

Información básica del dataset:
Filas: 5000, Columnas: 23

Primeras 5 filas:
  Student_ID First_Name Last_Name                    Email  Gender  Age  \
0      S1000       Omar  Williams  student0@university.com  Female   22   
1      S1001      Maria     Brown  student1@university.com    Male   18   
2      S1002      Ahmed     Jones  student2@university.com    Male   24   
3      S1003       Omar  Williams  student3@university.com  Female   24   
4      S1004       John     Smith  student4@university.com  Female   23   

    Department  Attendance (%)  Midterm_Score  Final_Score  Assignments_Avg  \
0  Mathematics           97.36          40.61        59.61            73.69   
1     Business           97.71          57.27        74.00            74.23   
2  Engineering           99.52          41.84        63.85            85.85   
3  Engineering           90.38          45.65        44.44            68.10   
4           CS           59.41          53.13        61.77            67.66   

   Quizzes_Avg  Participation_Score  Projects_Score  Total_Score Grade  \
0        53.17                 7.34           62.84        83.49     C   
1        98.23                 8.80           98.23        92.29     F   
2        50.00                 0.47           91.22        93.55     F   
3        66.27                 0.42           55.48        51.03     A   
4        83.98                 6.43           87.43        90.91     A   

   Study_Hours_per_Week Extracurricular_Activities Internet_Access_at_Home  \
0                  10.3                        Yes                      No   
1                  27.1                         No                      No   
2                  12.4                        Yes                      No   
3                  25.5                         No                     Yes   
4                  13.3                        Yes                      No   

  Parent_Education_Level Family_Income_Level  Stress_Level (1-10)  \
0               Master's              Medium                    1   
1            High School                 Low                    4   
2            High School                 Low                    9   
3            High School                 Low                    8   
4               Master's              Medium                    6   

   Sleep_Hours_per_Night  
0                    5.9  
1                    4.3  
2                    6.1  
3                    4.9  
4                    4.5  

Resumen estadístico:
               Age  Attendance (%)  Midterm_Score  Final_Score  \
count  5000.000000     5000.000000    5000.000000  5000.000000   
mean     21.048400       75.356076      70.701924    69.546552   
std       1.989786       14.392716      17.436325    17.108996   
min      18.000000       50.010000      40.000000    40.010000   
25%      19.000000       62.945000      55.707500    54.697500   
50%      21.000000       75.670000      70.860000    69.485000   
75%      23.000000       87.862500      85.760000    83.922500   
max      24.000000      100.000000      99.990000    99.980000   

       Assignments_Avg  Quizzes_Avg  Participation_Score  Projects_Score  \
count      5000.000000  5000.000000          5000.000000      5000.00000   
mean         74.956320    74.836214             4.996372        74.78305   
std          14.404287    14.423848             2.898978        14.54243   
min          50.000000    50.000000             0.000000        50.00000   
25%          62.340000    62.357500             2.507500        61.97000   
50%          75.090000    74.905000             4.960000        74.54000   
75%          87.352500    87.292500             7.550000        87.63000   
max          99.990000    99.990000            10.000000       100.00000   

       Total_Score  Study_Hours_per_Week  Stress_Level (1-10)  \
count  5000.000000           5000.000000          5000.000000   
mean     75.021860             17.521140             5.507200   
std      14.323246              7.193035             2.886662   
min      50.010000              5.000000             1.000000   
25%      62.710000             11.500000             3.000000   
50%      75.345000             17.400000             6.000000   
75%      87.060000             23.700000             8.000000   
max      99.990000             30.000000            10.000000   

       Sleep_Hours_per_Night  
count            5000.000000  
mean                6.514420  
std                 1.446155  
min                 4.000000  
25%                 5.300000  
50%                 6.500000  
75%                 7.800000  
max                 9.000000  

Tipos de datos y valores nulos:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 23 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   Student_ID                  5000 non-null   object 
 1   First_Name                  5000 non-null   object 
 2   Last_Name                   5000 non-null   object 
 3   Email                       5000 non-null   object 
 4   Gender                      5000 non-null   object 
 5   Age                         5000 non-null   int64  
 6   Department                  5000 non-null   object 
 7   Attendance (%)              5000 non-null   float64
 8   Midterm_Score               5000 non-null   float64
 9   Final_Score                 5000 non-null   float64
 10  Assignments_Avg             5000 non-null   float64
 11  Quizzes_Avg                 5000 non-null   float64
 12  Participation_Score         5000 non-null   float64
 13  Projects_Score              5000 non-null   float64
 14  Total_Score                 5000 non-null   float64
 15  Grade                       5000 non-null   object 
 16  Study_Hours_per_Week        5000 non-null   float64
 17  Extracurricular_Activities  5000 non-null   object 
 18  Internet_Access_at_Home     5000 non-null   object 
 19  Parent_Education_Level      3975 non-null   object 
 20  Family_Income_Level         5000 non-null   object 
 21  Stress_Level (1-10)         5000 non-null   int64  
 22  Sleep_Hours_per_Night       5000 non-null   float64
dtypes: float64(10), int64(2), object(11)
memory usage: 898.6+ KB
None

================================================================================
Análisis de correlación con la variable objetivo (Grade):
Total_Score              1.000000
Assignments_Avg          0.019396
Final_Score              0.017360
Age                      0.000375
Participation_Score     -0.001425
Midterm_Score           -0.002094
Quizzes_Avg             -0.005570
Stress_Level (1-10)     -0.007114
Attendance (%)          -0.009283
Study_Hours_per_Week    -0.009479
Sleep_Hours_per_Night   -0.017710
Projects_Score          -0.027344
dtype: float64

![image](https://github.com/user-attachments/assets/dbbb88b8-1754-4b56-a5d2-9fbf3fe3d7cc)
![image](https://github.com/user-attachments/assets/9d0c212f-feed-4052-8d9e-a63e9a38cd7e)
![image](https://github.com/user-attachments/assets/ad2f7919-0c32-45a6-8a1d-9b1ae7c33002)

================================================================================
Variables categóricas a codificar: Index(['Gender', 'Department', 'Grade', 'Extracurricular_Activities',
       'Internet_Access_at_Home', 'Parent_Education_Level',
       'Family_Income_Level'],
      dtype='object')

================================================================================
Evaluación del Modelo KNN con Distancia Personalizada:

Reporte de Clasificación:
              precision    recall  f1-score   support

           A       0.22      0.30      0.25       299
           B       0.19      0.21      0.20       304
           C       0.23      0.23      0.23       293
           D       0.20      0.16      0.18       303
           F       0.19      0.13      0.15       301

    accuracy                           0.21      1500
   macro avg       0.20      0.21      0.20      1500
weighted avg       0.20      0.21      0.20      1500


Matriz de Confusión:
![image](https://github.com/user-attachments/assets/e54ae657-1ad2-44a0-a573-206cc4f04ee5)

Precisión del modelo: 0.21

================================================================================
Optimización de Hiperparámetros:
Mejores parámetros: {'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
Mejor precisión en validación cruzada: 0.22

Evaluación del Modelo Optimizado:
              precision    recall  f1-score   support

           A       0.22      0.21      0.22       299
           B       0.17      0.18      0.17       304
           C       0.21      0.21      0.21       293
           D       0.24      0.25      0.24       303
           F       0.21      0.19      0.20       301

    accuracy                           0.21      1500
   macro avg       0.21      0.21      0.21      1500
weighted avg       0.21      0.21      0.21      1500

Precisión del modelo optimizado: 0.21

![image](https://github.com/user-attachments/assets/41eae6ba-5213-4da1-b469-38660442e0d3)

Desarrollado por: J.E. Carmona-Álvarez

            from sklearn.metrics import mean_squared_error
            
            # Evaluar diferentes valores de k
            error_rates = []
            mse_scores = []
            k_range = range(1, 21)
            
            for k in k_range:
                knn_model = KNeighborsClassifier(n_neighbors=k)
                knn_model.fit(X_train_scaled, y_train)
                y_pred_k = knn_model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred_k)
                mse = mean_squared_error(y_test, y_pred_k)
                error_rates.append(1 - acc)
                mse_scores.append(mse)
            
            # Visualización del Error y MSE
            plt.figure(figsize=(14, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(k_range, error_rates, marker='o', linestyle='--', color='red')
            plt.title('Error rate vs. K Value')
            plt.xlabel('Número de Vecinos (k)')
            plt.ylabel('Error de Clasificación')
            
            plt.subplot(1, 2, 2)
            plt.plot(k_range, mse_scores, marker='s', linestyle='-', color='blue')
            plt.title('MSE vs. K Value')
            plt.xlabel('Número de Vecinos (k)')
            plt.ylabel('Error Cuadrático Medio (MSE)')
            
            plt.tight_layout()
            plt.show()
            
            # Mostrar el mejor k en términos de menor MSE
            best_k_mse = k_range[np.argmin(mse_scores)]
            print(f"\nMejor valor de k según MSE: {best_k_mse}")
            print(f"MSE mínimo: {min(mse_scores):.4f}")
            
            print("")
            print("Desarrollado por: J.E. Carmona-Álvarez")

**Resultados:**
![image](https://github.com/user-attachments/assets/82ff2dc1-d32b-4837-b436-811784cd358c)

Mejor valor de k según MSE: 2
MSE mínimo: 3.9013

Desarrollado por: J.E. Carmona-Álvarez


















