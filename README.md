# ML
**TAREA 1: Ejercicio de rimas:**

_Enlace Google Colab: https://colab.research.google.com/drive/1phRzdGjHBnLQvY0ZUbRSVYL8eFfn9vgN?usp=sharing_

**1.1. Codigo para encontrar las palabras con las letras dadas:**

    # Libreria que se usa para expresiones regulares
    import re 
    
    archivo = "5000_palabras_comunes.txt"
    
    # ¿Cuáles son las letras con las que quiere buscar las palabras?
    # Mantener el símbolo $ al final de las letras y escribir todo en comillas
    # Ejm "ión$"
    terminacion = "das$"  
    
    # Función para leer las lineas del archivo
    with open(archivo, "r", encoding="utf-8") as f:
      palabras = f.read().splitlines()  # Lee todas las líneas y las convierte en una lista
    
    # Función para filtrar palabras que terminan con la terminación escogida
    palabras_filtradas = [palabra for palabra in palabras if re.search(terminacion, palabra)]
    
    # Mostrar las palabras encontradas
    print("Palabras que terminan con 'das':")
    print(palabras_filtradas)
    print("")
    print("Desarrollado por: J.E. Carmona-Álvarez")

**1.2. Aplicacion para subir archivo y buscar las palabras con la terminacion deseada:**

    import re
    from google.colab import files
    
    # Funcion para subir el archivo
    print("Por favor suba un archivo en formato .txt con una palabra por fila")
    archivo_subido = files.upload()
    nombre_archivo = list(archivo_subido.keys())[0]
    
    with open(nombre_archivo, "r", encoding="utf-8") as f:
        palabras = f.read().splitlines()  
    
    # Bucle para permitir intentos hasta encontrar palabras con la terminacion escogida
    while True:
        # Función para qeu el usuario vuelva a escribir la terminación
        terminacion = input("\nIngresa las letras con las que deben terminar las palabras (o escribe 'salir' para terminar): ")
    
        # Opción para salir del programa
        if terminacion.lower() == "salir":
            print("Programa terminado.")
            break
    
        # Expresión regular para encontrar palabras que terminan con la entrada del usuario
        patron = terminacion + "$"  # El signo $ indica "al final de la palabra"
        palabras_filtradas = [palabra for palabra in palabras if re.search(patron, palabra)]
    
        # Función para mostrar los resultados
        if palabras_filtradas:
            print("\nPalabras que terminan con '{}':".format(terminacion))
            for palabra in palabras_filtradas:
                print(palabra)
            break  
        else:
            print("\nNo se encontraron palabras que terminen con '{}'. Intenta con otra terminación.".format(terminacion))
            print("")
            print("Desarrollado por: J.E. Carmona-Álvarez")

**TAREA 2: De las siguientes librerías de Python describa algunos elementos que le hayan llamado la atención:**

2.1. **Numpy** 

2.2. **Pandas y Polars**

2.3. **Matplotlib y hvPlot**


**TAREA 3: Implemente utilizando Polars los siguientes algoritmos para encontrar reglas de asociación:**

_Enlace Google Colab: https://colab.research.google.com/drive/1PYjMSULj92htHh2cnyX_jBl4fihs4BEa?usp=sharing_

**3.1. Apriori**

        import polars as pl
        from efficient_apriori import apriori 
        
        # Datos de las transacciones
        transacciones = [
            ["Milk", "Bread", "Butter"],
            ["Milk", "Bread"],
            ["Bread", "Butter"],
            ["Milk", "Butter"],
            ["Milk", "Bread", "Butter"]
        ]
        
        # Aplicar Apriori
        conjuntos, reglas = apriori(transacciones, min_support=0.1, min_confidence=0.1)
        
        # Convertir conjuntos frecuentes a DataFrame de Polars
        df_conjuntos = pl.DataFrame(
            {
                "Ítems": [item for nivel in conjuntos.values() for item in nivel],  
                "Soporte": [conjuntos[nivel][item] for nivel in conjuntos for item in conjuntos[nivel]]  
            }
        )
        
        # Convertir reglas de asociación a DataFrame de Polars
        df_reglas = pl.DataFrame(
            {
                #Antecedente representa el ítem o conjunto que deben estar presentes en una transacción para que se dé una consecuencia
                "Antecedente": [list(rule.lhs) for rule in reglas], 
                #Consecuente de la regla es el ítem o conjunto de ítems que se predice que aparecerán en la transacción si se cumplen las condiciones del antecedente
                "Consecuente": [list(rule.rhs) for rule in reglas],  
                #Con que frecuencia aparece el consecuente en las transacciones que aparece el antecedente
                "Confianza": [rule.confidence for rule in reglas],
                #Qué tan frecuente es una regla
                "Soporte": [rule.support for rule in reglas],
                #Levantamiento de la regla: no es algo del azar 
                "Lift": [rule.lift for rule in reglas]
            }
        )
        
        # Filtrar reglas con antecedente de 2 elementos y consecuente de 1 elemento
        df_reglas_filtradas = df_reglas.filter(
            (df_reglas["Antecedente"].list.len() == 2) & 
            (df_reglas["Consecuente"].list.len() == 1)
        )
        
        # Mostrar resultados
        print("Conjuntos Frecuentes:")
        print(df_conjuntos)
        
        print("\n Reglas de Asociación:")
        print(df_reglas)
        
        print("\n Reglas Filtradas:")
        print(df_reglas_filtradas)
        print("")
        print("Desarrollado por: J.E. Carmona-Álvarez")

**3.2. FP-Growth**
       
        # Importamos las librerías necesarias
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        import pandas as pd
        from tabulate import tabulate
        
        # Transacciones
        transacciones = [
            ["Milk", "Bread", "Butter"],
            ["Milk", "Bread"],
            ["Bread", "Butter"],
            ["Milk", "Butter"],
            ["Milk", "Bread", "Butter"]
        ]
        
        # Convertimos las transacciones a formato adecuado para FP-Growth
        encoder = TransactionEncoder()
        encoded_array = encoder.fit(transacciones).transform(transacciones)
        df = pd.DataFrame(encoded_array, columns=encoder.columns_)
        
        # Calculamos los patrones frecuentes con FP-Growth, con soporte mínimo de 0.1
        frequent_itemsets = fpgrowth(df, min_support=0.1, use_colnames=True)
        
        # Calculamos las reglas de asociación
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.6)
        
        # Convertir los frozenset en listas de elementos (más legible)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
        
        # Seleccionamos las columnas relevantes: Antecedentes, Consecuentes, Confianza, Soporte y Lift
        result = rules[['antecedents', 'consequents', 'confidence', 'support', 'lift']]
        
        # Dar formato a la tabla utilizando tabulate
        formatted_result = tabulate(result, headers='keys', tablefmt='fancy_grid', showindex=False)
        
        # Imprimir la tabla formateada
        print(formatted_result)
              
        print("")
        print("Desarrollado por: J.E. Carmona-Álvarez")

**3.3. Compare ambos algoritmos con el mismo conjunto de datos**

_Resultados:_ Reglas de Asociación **Apriori**:
shape: (9, 5)

![image](https://github.com/user-attachments/assets/95577ff8-f24e-4ea2-bf58-1f6681d06f1c)

_Resultados:_ Reglas de asociación  **FP-Growth**:
shape: (12, 5)

![image](https://github.com/user-attachments/assets/b5ef7822-762c-4b22-8e33-1936144fb109)

**TAREA 4: Hopfield y PCA**
_Enlace Google Colab: https://colab.research.google.com/drive/1iepq3SXvebYm8RyG2JskDBIjOCC_orOZ?usp=sharing_

**4.1. Buscar el recorrido por todas las ciudades que demore menos tiempo, sin repetir ciudad utilizando redes de Hopfield**

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
--
El orden recomendado para visitar las ciudades en menos tiempo es: [3 2 4 0 1]

El tiempo mínimo para el recorriendo de estas ciudades es: 110 minutos

A=0, B=1, C=2, D=3, E=4

Desarrollado por: J.E. Carmona-Álvarez

**4.2. Utilizando PCA visualice en 2D una base de datos de MNIST**

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

![image](https://github.com/user-attachments/assets/d2fcca9a-830e-40f4-a529-9c9469a554f4)
