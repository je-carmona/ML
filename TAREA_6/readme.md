**TAREA 6: kNN**

_Enlace Google Colab:_https://colab.research.google.com/drive/13qG3ZuJTJoWz-HKcq-l0y6kX31Pt3sDx?usp=sharing

ENtrenamiento inicial del modelo para encontrar los vecinos cercanos con la base de datos:

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


Mejoramiento del rendimiento del codigo: 



**6.1.** How can you evaluate the hypothesis that the problem with almost all predictions giving `Medium` is due to the disproportionate data in that column?



