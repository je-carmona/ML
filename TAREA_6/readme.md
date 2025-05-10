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









Entrenamiento inicial del modelo para encontrar los vecinos cercanos con la base de datos:

      import pandas as pd
      import numpy as np
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler
      from sklearn.metrics import mean_squared_error
      import matplotlib.pyplot as plt
      from tqdm import tqdm  # Para mostrar progreso
      
      # Cargar los datos
      print("Cargando datos...")
      data = pd.read_csv('Students_Grading_Dataset.csv')
      
      # Preprocesamiento básico
      print("Preprocesando datos...")
      features = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 
                  'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 
                  'Projects_Score', 'Total_Score', 'Stress_Level (1-10)', 
                  'Sleep_Hours_per_Night']
      target = 'Study_Hours_per_Week'
      
      # Convertir variables categóricas de forma más segura
      data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Male' else 1)
      data['Extracurricular_Activities'] = data['Extracurricular_Activities'].apply(lambda x: 0 if x == 'No' else 1)
      data['Internet_Access_at_Home'] = data['Internet_Access_at_Home'].apply(lambda x: 0 if x == 'No' else 1)
      
      # Mapeo seguro de niveles educativos
      education_map = {'None': 0, 'High School': 1, 'Bachelor\'s': 2, 'Master\'s': 3, 'PhD': 4}
      data['Parent_Education_Level'] = data['Parent_Education_Level'].map(education_map).fillna(0).astype(int)
      
      # Mapeo seguro de ingresos
      income_map = {'Low': 0, 'Medium': 1, 'High': 2}
      data['Family_Income_Level'] = data['Family_Income_Level'].map(income_map).fillna(0).astype(int)
      
      # Añadir features adicionales
      features += ['Gender', 'Extracurricular_Activities', 'Internet_Access_at_Home',
                   'Parent_Education_Level', 'Family_Income_Level']
      
      # Limpieza de datos
      data = data.dropna(subset=features + [target])
      data = data.reset_index(drop=True)
      
      # Preparar X e y
      X = data[features]
      y = data[target]
      
      # Normalización
      print("Normalizando datos...")
      scaler = StandardScaler()
      X_normalized = scaler.fit_transform(X)
      
      # División train-test
      print("Dividiendo datos...")
      X_train, X_test, y_train, y_test = train_test_split(
          X_normalized, y, test_size=0.3, random_state=42)
      
      # Versión optimizada de KNN
      class OptimizedKNNRegressor:
          def __init__(self, k=5, metric='euclidean'):
              self.k = k
              self.metric = metric
              
          def fit(self, X, y):
              self.X_train = X
              self.y_train = y.values if hasattr(y, 'values') else y
              
          def predict(self, X):
              predictions = np.zeros(X.shape[0])
              for i, x in enumerate(X):
                  if self.metric == 'euclidean':
                      distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
                  elif self.metric == 'manhattan':
                      distances = np.sum(np.abs(self.X_train - x), axis=1)
                  elif self.metric == 'cosine':
                      norm_x = np.linalg.norm(x)
                      norm_train = np.linalg.norm(self.X_train, axis=1)
                      distances = 1 - np.dot(self.X_train, x) / (norm_train * norm_x)
                  else:
                      raise ValueError("Métrica no soportada")
                      
                  k_indices = np.argpartition(distances, self.k)[:self.k]
                  predictions[i] = np.mean(self.y_train[k_indices])
                  
              return predictions
      
      # Evaluación más rápida con menos valores de k
      print("Evaluando modelos...")
      k_values = range(1, 21, 2)  # Valores impares para evitar empates
      metrics = ['euclidean', 'manhattan', 'cosine']
      
      results = {}
      
      for metric in metrics:
          print(f"\nEvaluando métrica: {metric}")
          mse_scores = []
          for k in tqdm(k_values, desc=f'k values ({metric})'):
              knn = OptimizedKNNRegressor(k=k, metric=metric)
              knn.fit(X_train, y_train)
              y_pred = knn.predict(X_test)
              mse_scores.append(mean_squared_error(y_test, y_pred))
          
          results[metric] = mse_scores
          best_k = k_values[np.argmin(mse_scores)]
          print(f"Mejor k: {best_k} con MSE: {min(mse_scores):.4f}")
      
      # Gráfico de resultados
      plt.figure(figsize=(12, 6))
      for metric, mse_scores in results.items():
          plt.plot(k_values, mse_scores, 'o-', label=f'{metric} distance')
      
      plt.title('Error (MSE) vs. k para diferentes métricas')
      plt.xlabel('k (número de vecinos)')
      plt.ylabel('Error Cuadrático Medio (MSE)')
      plt.legend()
      plt.grid(True)
      plt.show()
      
      # Selección del mejor modelo
      best_metric = min(results, key=lambda k: min(results[k]))
      best_k = k_values[np.argmin(results[best_metric])]
      best_mse = min(results[best_metric])
      
      print(f"\nMejor combinación:")
      print(f"- Métrica: {best_metric}")
      print(f"- k: {best_k}")
      print(f"- MSE: {best_mse:.4f}")
      
      # Entrenar el mejor modelo
      print("\nEntrenando modelo final...")
      final_model = OptimizedKNNRegressor(k=best_k, metric=best_metric)
      final_model.fit(X_train, y_train)
      
      # Evaluación final
      y_pred = final_model.predict(X_test)
      final_mse = mean_squared_error(y_test, y_pred)
      final_mae = mean_absolute_error(y_test, y_pred)
      final_r2 = r2_score(y_test, y_pred)
      
      print("\nMétricas finales del modelo:")
      print(f"MSE: {final_mse:.4f}")
      print(f"MAE: {final_mae:.4f}")
      print(f"R²: {final_r2:.4f}")
      
      # Ejemplo de predicción
      print("\nEjemplo de predicciones vs valores reales:")
      sample_indices = np.random.choice(len(y_test), size=5, replace=False)
      for idx in sample_indices:
          print(f"Predicho: {y_pred[idx]:.2f} | Real: {y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]:.2f}")
      print("")
      print("Desarrollado por: J.E. Carmona-Álvarez")

Resultados: 
Evaluando métrica: euclidean
k values (euclidean): 100%|██████████| 10/10 [00:04<00:00,  2.22it/s]
Mejor k: 19 con MSE: 53.6838

Evaluando métrica: manhattan
k values (manhattan): 100%|██████████| 10/10 [00:05<00:00,  1.82it/s]
Mejor k: 19 con MSE: 53.8232

Evaluando métrica: cosine
k values (cosine): 100%|██████████| 10/10 [00:04<00:00,  2.43it/s]
Mejor k: 19 con MSE: 53.8000

![image](https://github.com/user-attachments/assets/fdcd61fb-dd4d-4279-b91a-781378a7fd20)

Mejor combinación:
- Métrica: euclidean
- k: 19
- MSE: 53.6838

Entrenando modelo final...

Métricas finales del modelo:
MSE: 53.6838
MAE: 6.2860
R²: -0.0582

Ejemplo de predicciones vs valores reales:
Predicho: 17.59 | Real: 25.00
Predicho: 18.13 | Real: 21.80
Predicho: 18.49 | Real: 26.40
Predicho: 14.92 | Real: 6.20
Predicho: 18.34 | Real: 15.30

Desarrollado por: J.E. Carmona-Álvarez

Mejoramiento del rendimiento del codigo: 

            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.preprocessing import RobustScaler
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.base import BaseEstimator, RegressorMixin
            import matplotlib.pyplot as plt
            from tqdm import tqdm
            from collections import defaultdict
            
            # Cargar y preprocesar datos
            print("Cargando y preprocesando datos...")
            data = pd.read_csv('Students_Grading_Dataset.csv')
            
            def preprocess_data(df):
                # Convertir variables categóricas
                df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
                df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'No': 0, 'Yes': 1})
                df['Internet_Access_at_Home'] = df['Internet_Access_at_Home'].map({'No': 0, 'Yes': 1})
                
                # Mapeo ordinal
                education_map = {'None': 0, 'High School': 1, 'Bachelor\'s': 2, 'Master\'s': 3, 'PhD': 4}
                df['Parent_Education_Level'] = df['Parent_Education_Level'].map(education_map).fillna(0).astype(int)
                
                income_map = {'Low': 0, 'Medium': 1, 'High': 2}
                df['Family_Income_Level'] = df['Family_Income_Level'].map(income_map).fillna(0).astype(int)
                
                # Crear nuevas características
                df['Academic_Performance'] = (df['Midterm_Score'] + df['Final_Score'] + df['Assignments_Avg']) / 3
                df['Stress_Sleep_Ratio'] = df['Stress_Level (1-10)'] / (df['Sleep_Hours_per_Night'] + 1e-6)  # Evitar división por cero
                df['Attendance_Score_Interaction'] = df['Attendance (%)'] * df['Academic_Performance']
                
                # Eliminar outliers
                for col in ['Study_Hours_per_Week', 'Academic_Performance', 'Total_Score']:
                    q1 = df[col].quantile(0.05)
                    q3 = df[col].quantile(0.95)
                    df = df[(df[col] >= q1) & (df[col] <= q3)]
                
                return df
            
            data = preprocess_data(data)
            
            # Selección de características
            features = [
                'Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg',
                'Academic_Performance', 'Stress_Sleep_Ratio', 'Attendance_Score_Interaction',
                'Gender', 'Parent_Education_Level', 'Family_Income_Level',
                'Stress_Level (1-10)', 'Sleep_Hours_per_Night'
            ]
            target = 'Study_Hours_per_Week'
            
            # Limpieza final
            data = data.dropna(subset=features + [target])
            data = data.reset_index(drop=True)
            
            # Preparar datos
            X = data[features]
            y = data[target]
            
            # Escalado robusto
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # División train-test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5))
            
            # Implementación mejorada de KNN compatible con scikit-learn
            class WeightedKNNRegressor(BaseEstimator, RegressorMixin):
                def __init__(self, k=5, metric='euclidean', weight=True):
                    self.k = k
                    self.metric = metric
                    self.weight = weight
                    
                def fit(self, X, y):
                    self.X_train = X
                    self.y_train = y.values if hasattr(y, 'values') else y
                    return self
                    
                def predict(self, X):
                    predictions = np.zeros(X.shape[0])
                    for i, x in enumerate(X):
                        if self.metric == 'euclidean':
                            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
                        elif self.metric == 'manhattan':
                            distances = np.sum(np.abs(self.X_train - x), axis=1)
                        elif self.metric == 'cosine':
                            norm_x = np.linalg.norm(x)
                            norm_train = np.linalg.norm(self.X_train, axis=1)
                            distances = 1 - np.dot(self.X_train, x) / (norm_train * norm_x)
                        else:
                            raise ValueError("Métrica no soportada")
                            
                        k_indices = np.argpartition(distances, self.k)[:self.k]
                        k_distances = distances[k_indices]
                        k_values = self.y_train[k_indices]
                        
                        if self.weight:
                            weights = 1 / (k_distances + 1e-6)  # Suavizado
                            predictions[i] = np.sum(weights * k_values) / np.sum(weights)
                        else:
                            predictions[i] = np.mean(k_values)
                            
                    return predictions
                
                def get_params(self, deep=True):
                    return {'k': self.k, 'metric': self.metric, 'weight': self.weight}
                
                def set_params(self, **parameters):
                    for parameter, value in parameters.items():
                        setattr(self, parameter, value)
                    return self
            
            # Optimización de hiperparámetros
            print("\nOptimizando hiperparámetros...")
            param_grid = {
                'k': range(1, 31, 2),  # Valores impares
                'metric': ['euclidean', 'manhattan'],
                'weight': [True, False]
            }
            
            best_score = float('inf')
            best_params = {}
            results = defaultdict(list)
            
            # Búsqueda en grilla manual con validación cruzada
            for k in tqdm(param_grid['k'], desc='Evaluando k'):
                for metric in param_grid['metric']:
                    for weight in param_grid['weight']:
                        model = WeightedKNNRegressor(k=k, metric=metric, weight=weight)
                        
                        # Validación cruzada más rápida con menos folds
                        try:
                            scores = -cross_val_score(model, X_train, y_train, 
                                                    cv=3, scoring='neg_mean_squared_error',
                                                    n_jobs=-1)
                            avg_score = np.mean(scores)
                            
                            results['k'].append(k)
                            results['metric'].append(metric)
                            results['weight'].append(weight)
                            results['mse'].append(avg_score)
                            
                            if avg_score < best_score:
                                best_score = avg_score
                                best_params = {'k': k, 'metric': metric, 'weight': weight}
                        except:
                            continue
            
            # Resultados de la búsqueda
            results_df = pd.DataFrame(results)
            print("\nTop 5 combinaciones de parámetros:")
            print(results_df.sort_values('mse').head(5))
            
            print(f"\nMejores parámetros: {best_params}")
            print(f"MSE promedio en validación cruzada: {best_score:.4f}")
            
            # Entrenar modelo final
            final_model = WeightedKNNRegressor(**best_params)
            final_model.fit(X_train, y_train)
            
            # Evaluación final
            y_pred = final_model.predict(X_test)
            final_mse = mean_squared_error(y_test, y_pred)
            final_mae = mean_absolute_error(y_test, y_pred)
            final_r2 = r2_score(y_test, y_pred)
            
            print("\nMétricas finales en conjunto de prueba:")
            print(f"MSE: {final_mse:.4f}")
            print(f"MAE: {final_mae:.4f}")
            print(f"R²: {final_r2:.4f}")
            
            # Visualización
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
            plt.title('Predicciones vs Valores Reales')
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            
            plt.subplot(1, 2, 2)
            errors = y_test - y_pred
            plt.hist(errors, bins=30)
            plt.title('Distribución de Errores')
            plt.xlabel('Error (Real - Predicho)')
            plt.suptitle(f'KNN Regressor (k={best_params["k"]}, metric={best_params["metric"]})')
            plt.tight_layout()
            plt.show()
            print("")
            print("Desarrollado por: J.E. Carmona-Álvarez")

Resultados: 

Optimizando hiperparámetros...
Evaluando k: 100%|██████████| 15/15 [00:33<00:00,  2.26s/it]

Top 5 combinaciones de parámetros:
     k     metric  weight        mse
57  29  euclidean   False  43.455829
56  29  euclidean    True  43.515423
53  27  euclidean   False  43.534223
59  29  manhattan   False  43.561216
55  27  manhattan   False  43.563866

Mejores parámetros: {'k': 29, 'metric': 'euclidean', 'weight': False}
MSE promedio en validación cruzada: 43.4558

Métricas finales en conjunto de prueba:
MSE: 43.2106
MAE: 5.6570
R²: -0.0497

![image](https://github.com/user-attachments/assets/68eff768-d0e0-43e0-bbc5-6ca56563322a)

Desarrollado por: J.E. Carmona-Álvarez





