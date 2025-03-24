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
    #permite usar los archivos en el entorno de Colab
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

Las librerias de Numpy, Panda, Polars, Matplotlib y hvPlot sonfrecuentemente usadas en el análisis de datos y ciencias de datos

2.1. **Numpy** 

La libreria de Numpy tiene una ventaja significativa en el caluo de expresiones numericas y operaciones con matrices. Dentro de los algoritmos de la libreria se encuentra la funcion _ndarray_, que permiten almacenar y manipular grandes volúmenes de datos de manera eficiente. La libreria de Numpy por medio de sus algoritmos tambien permite realizar operaciones entre arrays de diferentes formas de manera eficiente, sin necesidad de escribir bucles explícitos lo que simplifica y acelera muchas operaciones en manejo de análisis y ciencia de datos. _Fuente: https://aprendeconalf.es/docencia/python/manual/numpy/_

2.2. **Pandas y Polars**

Pndas sirve para el manejo y análsisi de datos en forma de tabla o matrices,por otra parte Pandas tiene un algoritmo para la estructura de datos conocido como DataFrame, que se puede interpretar como una tabla bidimensional con la que se puede fácilmente manejar datos estructurados. El DataFrame permite seleccionar, filtrar, transformar y agregar datos de manera intuitiva. Pandas incluye métodos robustos para gestionar valores faltantes, como NaN (Not a Number), y permite rellenar, eliminar o interpolar estos valores de manera sencilla.

Polar sirve para el manejo de conjuntos multiples bases de datos, hacer tareas de análisis, administracion e procesamiento de datos complejos. La libreria de Polars es una de las mas recientes y ha sido diseñada para ser mas eficiente en el procesamiento que la libreri de Pandas, especialmente en grandes volúmenes de datos. Está optimizada para realizar operaciones sobre DataFrames de manera muy eficiente utilizando múltiples hilos de procesamiento. 

Fuente: https://www.vernegroup.com/actualidad/tecnologia/introduccion-polars-tratamiento-datos-comparativa-pandas/
Fuente: https://www.youtube.com/watch?v=E2Ki0Wd9cL0

2.3. **Matplotlib y hvPlot**

La libreria de Matplotlib es extremadamente flexible para crear gráficos estáticos en Python ya que permite personalizar gráficos segun su tipo de gráfico hasta los detalles más pequeños, como las etiquetas, leyendas y colores. Por su parte la libreria de hvPlot está diseñado para crear visualizaciones interactivas de manera muy sencilla, ya que proporciona una interfaz asertiva sobre Pandas, permitiendo generar gráficos interactivos con solo unas pocas líneas de código.

Fuente: https://www.youtube.com/watch?v=AVf0SDXNowo


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

**TAREA 5: Decision Tree**

_Enlace Google Colab: https://colab.research.google.com/drive/17FWE7lUU_2tmAOlBweKxPSowAGShuSRq?usp=sharing_

**5.1. Make the Decision Tree algorithm for categories**

        #Manejo de arreglos y matrices numéricas
        import numpy as np
        
        #Sk Learn Algoritmos de Machine Learning, preprocesamiento y métricas
        #Convierte datos categóricos en números
        from sklearn.preprocessing import LabelEncoder
        #Divide los datos en entrenamiento y prueba
        from sklearn.model_selection import train_test_split
        #Modelo de Árbol de Decisión
        from sklearn.tree import DecisionTreeClassifier
        #Mide la precisión del modelo
        from sklearn.metrics import accuracy_score
        #Muestra métricas de rendimiento
        from sklearn.metrics import classification_report
        
        #Graficar el árbol de decisión
        import matplotlib.pyplot as plt
        #Dibuja el árbol de decisión
        from sklearn.tree import plot_tree
        #Tiene algoritmos de herramientas estadísticas avanzadas
        from scipy.stats import entropy
        
        # Datos en formato numpy
        data = np.array([
            ["G", "G", "R", "E"],
            ["R", "G", "B", "M"],
            ["B", "R", "G", "A"],
            ["G", "R", "G", "E"],
            ["R", "B", "R", "A"],
            ["G", "G", "G", "E"],
            ["B", "G", "R", "M"],
            ["R", "R", "R", "M"],
            ["G", "B", "G", "A"],
            ["B", "B", "B", "A"]
        ])
        
        # Separar características (X) y variable objetivo (y)
        # Las 3 primeras columnas representan las caliicaciones de las asignaturas Matemáticas (Ma), Ciencias (Sc), e Ingles (En)
        X = data[:, :-1]  
        # La última columna representa las preferencias de las carreras (Pc)
        y = data[:, -1]   
        
        # Cálculo de la entropía antes de construir el árbol
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        H = entropy(probabilities, base=2)
        print("Entropía del conjunto de datos:", H)
        print("")
        
        # Codificar datos categóricos a números correctamente
        label_encoders = [LabelEncoder() for _ in range(X.shape[1])]  # Un encoder por cada columna
        X_encoded = np.array([le.fit_transform(X[:, i]) for i, le in enumerate(label_encoders)]).T
        
        # Codificar la variable objetivo
        encoder_y = LabelEncoder()
        y_encoded = encoder_y.fit_transform(y)
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)
        
        # Entrenar el modelo
        clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        clf.fit(X_train, y_train)
        
        # Visualizar el árbol de decisión
        plt.figure(figsize=(14, 6))
        plot_tree(clf, feature_names=["Ma", "Sc", "En"], class_names=encoder_y.classes_, filled=True)
        plt.show()
        
        print("")
        print("Desarrollado por: J.E. Carmona-Álvarez")

![image](https://github.com/user-attachments/assets/f06202e4-aee7-4538-8efd-58b8fd9e75df)

**5.2. Explore the scikit-learn algorithms**

El algoritmo _Scikit-learn_ es una libreria de Python que contiene algoritmos para aprendizaje automatizado (_Machine Learning_) algunas de las principales caracteristicas de esta libreria se agrupan como: 

 * **Clasificación:** Para identificar a qué categoría pertenece un objeto, algunas aplicaciones de clasiicacion se basan en detección de spam y reconocimiento de imágenes. Lo algoritmos representativos de esta categoria son aumento de gradiente, vecino cercano, regresion logística. 
 * **Regresión:** seusa para la predicción de un atributo de valor continuo asociado a un objeto, algunas de las aplicaciones mas comúnes son la respuesta a medicamentos y precios de las acciones. Los algoritmos representativos de esta clasificación son el aumento de gradiente, vecinos más cercanos, bosque aleatorio, cresta, entre otros.
 * **Agrupamiento:** funciona para la agrupación automática de objetos similares en conjuntos, se aplica en la segmentación de clientes, agrupación de resultados de experimentos. Los algoritmos mas comunes son k-Means, HDBSCAN y agrupamiento jerárquico
 * **Reducción de la dimensionalidad:** Se usa para reducir el número de variables aleatorias a tener en cuenta. Sus aplicaciones se fundamentan en las visualización y aumento de la eficiencia. Los algoritmos ampliamente conocidos de esta categoria son PCA, selección de características y factorización de matrices no negativas
 * **Selección de modelos:** Esta categoria sirve para la comparación, validación y elección de parámetros y modelos. Se aplica en precisión mejorada a través del ajuste de parámetros, y los algoritmos mas comunes son la búsqueda en cuadrícula, validación cruzada y métricas.
 * **Preprocesamiento:** Esta caracteristica sirve para la extracción y normalización de características, los aplicaciones mas típicas son la transformación de datos de entrada, como texto, para su uso se implementan principalmente los algoritmos de aprendizaje automático. Los algoritmos diseñados para esta caracteristica son el preprocesamiento y extracción de características.

Tomado de: https://scikit-learn.org/stable/index.html

Para consultar todos los algoritmos de la libreria _Scikit-learn_ se puede ejecutar el siguiente código: 
        
        from sklearn.utils import all_estimators
        
        modelos = all_estimators()
        print([modelo[0] for modelo in modelos])


 _**5.2.1. DecisionTreeClassifier**_


        import numpy as np
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from sklearn.preprocessing import LabelEncoder
        import matplotlib.pyplot as plt
        
        # Datos originales
        data = [
            ["G", "G", "R", "E"],
            ["R", "G", "B", "M"],
            ["B", "R", "G", "A"],
            ["G", "R", "G", "E"],
            ["R", "B", "R", "A"],
            ["G", "G", "G", "E"],
            ["B", "G", "R", "M"],
            ["R", "R", "R", "M"],
            ["G", "B", "G", "A"],
            ["B", "B", "B", "A"]
        ]
        
        # Convertir los datos en un DataFrame
        df = pd.DataFrame(data, columns=["Matemáticas", "Ciencias", "Ingles", "Carrera"])
        
        # Usar LabelEncoder para convertir las letras en números
        le = LabelEncoder()
        
        # Aplicamos el LabelEncoder a todas las columnas excepto la de clase
        df["Matemáticas"] = le.fit_transform(df["Matemáticas"])
        df["Ciencias"] = le.fit_transform(df["Ciencias"])
        df["Ingles"] = le.fit_transform(df["Ingles"])
        df["Carrera"] = le.fit_transform(df["Carrera"])
        
        # Separar las características y la clase
        X = df[["Matemáticas", "Ciencias", "Ingles"]]  # Características
        y = df["Carrera"]  # Clase
        
        # El algoritmo sklearn por defecto crea y entrenar el árbol de decisión por el metodo de GINI
        # Para usar el metodo de Entropy se usa la expresión 'DecisionTreeClassifier(criterion="entropy", random_state=42)'
        #clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        #clf.fit(X, y)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        
        # Graficar el árbol de decisión
        plt.figure(figsize=(18, 6))
        plot_tree(clf, feature_names=["Matemáticas", "Ciencias", "Ingles"], class_names=le.classes_, filled=True)
        
        # Guardar la imagen
        plt.savefig('decision_tree_entropy.png')
        
        # Mostrar la imagen
        plt.show()
        
        print("")
        print("Desarrollado por: J.E. Carmona-Álvarez")

Por el metodo de gini 

![image](https://github.com/user-attachments/assets/db22072a-877a-4222-ab71-5195ecfe4161)


Por el metodo de Entropy

![image](https://github.com/user-attachments/assets/03bbeb1b-68c0-4c08-966b-4cdd2107b9f6)

 
 _**5.2.2. RandomForestClassifier**_

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.tree import plot_tree
        
        # Datos originales
        data = [
            ["G", "G", "R", "E"],
            ["R", "G", "B", "M"],
            ["B", "R", "G", "A"],
            ["G", "R", "G", "E"],
            ["R", "B", "R", "A"],
            ["G", "G", "G", "E"],
            ["B", "G", "R", "M"],
            ["R", "R", "R", "M"],
            ["G", "B", "G", "A"],
            ["B", "B", "B", "A"]
        ]
        
        # Convertir los datos en un DataFrame
        df = pd.DataFrame(data, columns=["Matemáticas", "Ciencias", "Inglés", "Carrera"])
        
        # Codificar las variables categóricas en números
        le = LabelEncoder()
        for col in df.columns:
            df[col] = le.fit_transform(df[col])
        
        # Separar características y clase
        X = df[["Matemáticas", "Ciencias", "Inglés"]]
        y = df["Carrera"]
        
        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Crear el modelo RandomForestClassifier con entropía
        clf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=42)
        clf.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = clf.predict(X_test)
        
        # Evaluar la precisión del modelo
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Precisión del modelo: {accuracy:.2f}')
        
        # Visualizar un árbol dentro del bosque
        plt.figure(figsize=(16, 8))
        plot_tree(clf.estimators_[0], feature_names=["Matemáticas", "Ciencias", "Inglés"], class_names=["Artes", "Ingeniería", "Medicina"], filled=True)
        plt.title("Árbol de decisión dentro del Bosque Aleatorio (Entropía)")
        plt.show()
        
        print("")
        print("Desarrollado por: J.E. Carmona-Álvarez")
        
![image](https://github.com/user-attachments/assets/29f5986b-af7c-4e53-9a8c-f371c9fba7db)

**5.3. Presentation on Supervised Learning with which you have previous experience**

Clasificación del **Estado Post-Evento** de los elementos físicos del sistema de infraestructura afectado por desatres provocados por fenomenos naturles, en los departamentos de:
* AMAZONAS
* ANTIOQUIA
* BOYACA
* CORDOBA
* HUILA
* QUINDIO
* TOLIMA

**MATRIZ DE DATOS**

![image](https://github.com/user-attachments/assets/08575289-50b5-467a-ba61-82ca53263d0b)

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.tree import plot_tree
        
        # Datos
        data = [
            ["Parcialmente funcional", "Operacional", "Operacional", "Baja Resiliencia"],
            ["Parcialmente funcional", "No funcional", "No funcional", "Alta Resiliencia"],
            ["Operacional", "Parcialmente funcional", "Parcialmente funcional", "Alta Resiliencia"],
            ["No funcional", "Operacional", "Operacional", "Media Resiliencia"],
            ["Parcialmente funcional", "No funcional", "No funcional", "Media Resiliencia"],
            ["Operacional", "No funcional", "Operacional", "Alta Resiliencia"],
            ["Operacional", "Parcialmente funcional", "No funcional", "Alta Resiliencia"]
        ]
        
        # Convertir en DataFrame
        df = pd.DataFrame(data, columns=["AVALANCHA", "DESLIZAMIENTO", "INUNDACION", "HIPOTESIS"])
        
        # Codificar variables categóricas
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in df.columns:
            df_encoded[col] = le.fit_transform(df[col])
        
        # Separar características y clase
        X = df_encoded[["AVALANCHA", "DESLIZAMIENTO", "INUNDACION"]]
        y = df_encoded["HIPOTESIS"]
        
        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modelo RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=42)
        clf.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = clf.predict(X_test)
        
        # Evaluar precisión
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Precisión del modelo: {accuracy:.2f}')
        
        # Visualizar un árbol dentro del bosque
        plt.figure(figsize=(14, 10))
        plot_tree(clf.estimators_[0], feature_names=["AVALANCHA", "DESLIZAMIENTO", "INUNDACION"], 
                  class_names=["Baja Resiliencia", "Media Resiliencia", "Alta Resiliencia"], filled=True)
        plt.title("Árbol de decisión dentro del Bosque Aleatorio (Entropía)")
        plt.show()
        
        print("\nDesarrollado por: J.E. Carmona-Álvarez")

![image](https://github.com/user-attachments/assets/ddd0f2e7-a01d-4216-ae76-5d0c77368b2d)

Clasificación de los **Tipos de Daño** en los elementos físicos del sistema de infraestructura afectado por desatres provocados por fenomenos naturles, en los departamentos de:
* AMAZONAS
* ANTIOQUIA
* BOYACA
* CORDOBA
* HUILA
* QUINDIO
* TOLIMA

**MATRIZ DE DATOS**

![image](https://github.com/user-attachments/assets/650a17cb-a1af-452a-9eee-11e88a0c3f12)

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.tree import plot_tree
        
        # Datos
        data = [
            ["Daño moderado", "Sin daño", "Sin daño", "Baja Resiliencia"],
            ["Daño moderado", "Daño severo", "Daño severo", "Alta Resiliencia"],
            ["Sin daño", "Daño moderado", "Daño moderado", "Alta Resiliencia"],
            ["Daño severo", "Sin daño", "Sin daño", "Media Resiliencia"],
            ["Daño moderado", "Daño severo", "Daño severo", "Media Resiliencia"],
            ["Sin daño", "Daño severo", "Sin daño", "Alta Resiliencia"],
            ["Sin daño", "Daño moderado", "Daño severo", "Alta Resiliencia"]
        ]
        
        # Convertir en DataFrame
        df = pd.DataFrame(data, columns=["AVALANCHA", "DESLIZAMIENTO", "INUNDACION", "HIPOTESIS"])
        
        # Codificar variables categóricas
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in df.columns:
            df_encoded[col] = le.fit_transform(df[col])
        
        # Separar características y clase
        X = df_encoded[["AVALANCHA", "DESLIZAMIENTO", "INUNDACION"]]
        y = df_encoded["HIPOTESIS"]
        
        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Modelo RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=42)
        clf.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = clf.predict(X_test)
        
        # Evaluar precisión
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Precisión del modelo: {accuracy:.2f}')
        
        # Visualizar un árbol dentro del bosque
        plt.figure(figsize=(10 , 6))
        plot_tree(clf.estimators_[0], feature_names=["AVALANCHA", "DESLIZAMIENTO", "INUNDACION"], 
                  class_names=le.inverse_transform([0, 1, 2]), filled=True)
        plt.title("Árbol de decisión dentro del Bosque Aleatorio (Entropía)")
        plt.show()
        
        print("\nDesarrollado por: J.E. Carmona-Álvarez")


![image](https://github.com/user-attachments/assets/25f5a3ae-4adf-4099-9880-ec1d14b8f7bd)

