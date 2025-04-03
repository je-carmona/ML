TAREA 4: Hopfield y PCA

Enlace Google Colab: https://colab.research.google.com/drive/1iepq3SXvebYm8RyG2JskDBIjOCC_orOZ?usp=sharing

4.1. Buscar el recorrido por todas las ciudades que demore menos tiempo, sin repetir ciudad utilizando redes de Hopfield

    import numpy as np
    
    # Número de ciudades y matriz de distancias
    n_ciudades = 5
    distancias = np.array([
        [0, 5, 5, 6, 4],
        [5, 0, 3, 7, 8],
        [5, 3, 0, 4, 8],
        [6, 7, 4, 0, 5],
        [4, 8, 8, 5, 0]
    ])
    
    # Parámetros de la red de Hopfield
    def calcular_energia(solucion, distancias):
        energia = 0
        for i in range(len(solucion)):
            for j in range(len(solucion)):
                energia += distancias[solucion[i], solucion[j]]
        return energia
    
    # Simulación de Hopfield
    def red_hopfield(n_ciudades, distancias, iteraciones=1000):
        solucion_actual = np.random.permutation(n_ciudades)  # Inicializar con una permutación aleatoria
        mejor_solucion = solucion_actual.copy()
        mejor_energia = calcular_energia(mejor_solucion, distancias)
    
        for _ in range(iteraciones):
            # Cambiar dos ciudades aleatoriamente
            idx1, idx2 = np.random.choice(n_ciudades, 2, replace=False)
            solucion_actual[idx1], solucion_actual[idx2] = solucion_actual[idx2], solucion_actual[idx1]
    
            energia_actual = calcular_energia(solucion_actual, distancias)
            if energia_actual < mejor_energia:
                mejor_energia = energia_actual
                mejor_solucion = solucion_actual.copy()
    
        return mejor_solucion, mejor_energia
    
    # Ejecutar la red de Hopfield
    mejor_recorrido, mejor_tiempo = red_hopfield(n_ciudades, distancias)
    print("Mejor recorrido:", mejor_recorrido)
    print("Tiempo mínimo:", mejor_tiempo)
    
    print("")
    print("Desarrollado por: J.E. Carmona-Álvarez")
-- El orden recomendado para visitar las ciudades en menos tiempo es: [3 2 4 0 1]

El tiempo mínimo para el recorriendo de estas ciudades es: 110 minutos

A=0, B=1, C=2, D=3, E=4

Desarrollado por: J.E. Carmona-Álvarez

4.2. Utilizando PCA visualice en 2D una base de datos de MNIST

    #Librerias
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.datasets import fetch_openml
    
    # Cargar los datos de MNIST desde OpenML
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    # Extraer las imágenes y etiquetas
    X = mnist.data  # Matriz de características (70,000 imágenes de 784 píxeles cada una)
    y = mnist.target.astype(int)  # Etiquetas de los dígitos
    
    # Aplicar PCA con 2 componentes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Definir una paleta de colores para los dígitos
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.5, s=10)
    plt.colorbar(scatter, label="Dígitos")
    plt.xlabel("Componente Principal 1 -  Dirección de máxima varianza")
    plt.ylabel("Componente Principal 2 - Dirección perpendicular a PC1")
    plt.title("VISUALIZACIÓN DE MNIST EN 2D USANDO PCA")
    plt.show()
    
    print("")
    print("Desarrollado por: J.E. Carmona-Álvarez")

![image](https://github.com/user-attachments/assets/e86a4ad2-5442-4dd0-901c-6cf9fbd0cd61)
