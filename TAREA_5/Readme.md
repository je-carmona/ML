**TAREA 5: Árbol de decisión**

_Enlace Google Colab:_ https://colab.research.google.com/drive/17FWE7lUU_2tmAOlBweKxPSowAGShuSRq?usp=sharing

**5.1.** Hacer el algoritmo de árbol de decisión para categorías

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

![image](https://github.com/user-attachments/assets/02b699b1-a53f-4385-850c-b4b5ff91b503)

**5.2.** Explora los algoritmos de scikit-learn

El algoritmo Scikit-learn es una librería de Python que contiene algoritmos para aprendizaje automatizado (Machine Learning), algunas de las principales características de esta librería se agrupan como:

Clasificación: Para identificar a qué categoría pertenece un objeto, algunas aplicaciones de clasificación se basan en detección de spam y reconocimiento de imágenes. Lo algoritmos representativos de esta categoría son aumento de gradiente, vecino cercano, regresión logística.
Regresión: seusa para la predicción de un atributo de valor continuo asociado a un objeto, algunas de las aplicaciones más comunes son la respuesta a medicamentos y precios de las acciones. Los algoritmos representativos de esta clasificación son el aumento de gradiente, vecinos más cercanos, bosque aleatorio, cresta, entre otros.
Agrupamiento: funciona para la agrupación automática de objetos similares en conjuntos, se aplica en la segmentación de clientes, agrupación de resultados de experimentos. Los algoritmos más comunes son k-Means, HDBSCAN y agrupamiento jerárquico
Reducción de la dimensionalidad: Se usa para reducir el número de variables aleatorias a tener en cuenta. Sus aplicaciones se fundamentan en las visualización y aumento de la eficiencia. Los algoritmos ampliamente conocidos de esta categoría son PCA, selección de características y factorización de matrices no negativas
Selección de modelos: Esta categoría sirve para la comparación, validación y elección de parámetros y modelos. Se aplica en precisión mejorada a través del ajuste de parámetros, y los algoritmos más comunes son la búsqueda en cuadrícula, validación cruzada y métricas.
Preprocesamiento: Esta característica sirve para la extracción y normalización de características, las aplicaciones más típicas son la transformación de datos de entrada, como texto, para su uso se implementan principalmente los algoritmos de aprendizaje automático. Los algoritmos diseñados para esta característica son el preprocesamiento y extracción de características.
Tomado de: https://scikit-learn.org/stable/index.html

Para consultar todos los algoritmos de la librería Scikit-learn se puede ejecutar el siguiente código:

    from sklearn.utils import all_estimators
    
    modelos = all_estimators()
    print([modelo[0] for modelo in modelos])

**5.2.1.** DecisionTreeClassifier

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
![image](https://github.com/user-attachments/assets/7a311b77-0839-4149-a069-47c31a30ac09)

Por el método de Entropy
![image](https://github.com/user-attachments/assets/967df628-a2a4-46a0-baa2-75bba9322ce7)


**5.2.2.** Clasificador aleatorio de ForestForestClasificador

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

    ![image](https://github.com/user-attachments/assets/0d8bc8c7-23f2-460f-af01-a80b9e4d7e79)


**5.3.** Presentación sobre el Aprendizaje Supervisado con el que tienes experiencia previa

Clasificación del Estado Post-Evento de los elementos físicos del sistema de infraestructura afectado por desastres provocados por fenomenos naturles, en los departamentos de:

AMAZONAS
ANTIOQUIA
BOYACA
CÓRDOBA
HUILA
QUINDÍO
TOLIMA

**MATRIZ DE DATOS**

![image](https://github.com/user-attachments/assets/95d18dfb-dd14-428b-b424-b194d552b9e2)

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

Clasificación de los Tipos de Daño en los elementos físicos del sistema de infraestructura afectado por desatres provocados por fenomenos naturles, en los departamentos de:

AMAZONAS
ANTIOQUIA
BOYACA
CÓRDOBA
HUILA
QUINDÍO
TOLIMA

**MATRIZ DE DATOS**

![image](https://github.com/user-attachments/assets/c77afaf2-821a-4387-b85a-9a9ee318a500)

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
    
![image](https://github.com/user-attachments/assets/25cb6219-a0f4-4c26-abc4-feee328359fd)



