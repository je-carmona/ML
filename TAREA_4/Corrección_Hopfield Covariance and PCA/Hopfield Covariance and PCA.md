**Corrección Tarea 4:** Hopfield Covariance and PCA

_Enlace Colab:_ [https://colab.research.google.com/drive/1iepq3SXvebYm8RyG2JskDBIjOCC_orOZ?usp=sharing](url)  

      #CORRECCIÓN - Hopfield Covariance and PCA
      
      import numpy as np
      import itertools
      import matplotlib.pyplot as plt
      from sklearn.decomposition import PCA
      
      
      # 1. Datos del problema
      
      n_ciudades = 5
      distancias = np.array([
          [0, 5, 5, 6, 4],
          [5, 0, 3, 7, 8],
          [5, 3, 0, 4, 8],
          [6, 7, 4, 0, 5],
          [4, 8, 8, 5, 0]
      ])
      ciudades = ['A', 'B', 'C', 'D', 'E']
      
      
      # 2. Solución Óptima (Fuerza Bruta)
      
      def tsp_brute_force(distancias):
          n = distancias.shape[0]
          indices = list(range(n))
          mejor_ruta = None
          minima_distancia = float('inf')
          
          for permutacion in itertools.permutations(indices):
              distancia = 0
              for i in range(n):
                  distancia += distancias[permutacion[i], permutacion[(i + 1) % n]]
              if distancia < minima_distancia:
                  minima_distancia = distancia
                  mejor_ruta = permutacion
          return mejor_ruta, minima_distancia
      
      ruta_optima, dist_optima = tsp_brute_force(distancias)
      print("Ruta óptima:", [ciudades[i] for i in ruta_optima], "| Distancia:", dist_optima)
      
      
      # 3. Red de Hopfield (Ajustada)
      
      def hopfield_tsp(distancias, iteraciones=1000, A=10, B=10, C=0.1):
          n = distancias.shape[0]
          V = np.random.rand(n, n)
          
          for _ in range(iteraciones):
              for x in range(n):
                  for i in range(n):
                      sum_col = np.sum(V[:, i]) - V[x, i]
                      sum_row = np.sum(V[x, :]) - V[x, i]
                      sum_dist = 0
                      for y in range(n):
                          if y != x:
                              next_i = (i + 1) % n
                              prev_i = (i - 1) % n
                              sum_dist += distancias[x, y] * (V[y, next_i] + V[y, prev_i])
                      du = -A * (sum_row - 1) - B * (sum_col - 1) - C * sum_dist
                      V[x, i] = 1 / (1 + np.exp(-du))
          
          # Interpretar la solución
          ruta = np.argmax(V, axis=1)
          distancia = sum(distancias[ruta[i], ruta[(i + 1) % n]] for i in range(n))
          return ruta, distancia
      
      ruta_hopfield, dist_hopfield = hopfield_tsp(distancias)
      print("Ruta Hopfield:", [ciudades[i] for i in ruta_hopfield], "| Distancia:", dist_hopfield)
      
      
      # 4. Visualización con PCA
      
      pca = PCA(n_components=2)
      coords_2d = pca.fit_transform(distancias)
      
      plt.figure(figsize=(10, 5))
      plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c='red', s=200)
      for i, ciudad in enumerate(ciudades):
          plt.text(coords_2d[i, 0], coords_2d[i, 1], ciudad, fontsize=12)
      
      # Dibujar rutas
      def plot_ruta(ruta, color, label):
          ruta_cerrada = np.append(ruta, ruta[0])
          plt.plot(coords_2d[ruta_cerrada, 0], coords_2d[ruta_cerrada, 1], color, label=label)
      
      plot_ruta(ruta_optima, 'g-', 'Óptima')
      plot_ruta(ruta_hopfield, 'b--', 'Hopfield')
      plt.legend()
      plt.title("Comparación de Rutas TSP")
      plt.grid()
      plt.show()
      print("")
      print("Desarrollado por: J.E. Carmona-Álvarez") 

**Resultado:** 

Ruta óptima: ['A', 'B', 'C', 'D', 'E'] | Distancia: 21
Ruta Hopfield: ['E', 'D', 'B', 'A', 'C'] | Distancia: 30

![image](https://github.com/user-attachments/assets/88f8d07f-390f-4cd2-a81b-41a7c73bb248)

Desarrollado por: J.E. Carmona-Álvarez
      
