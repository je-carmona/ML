# ML
**TAREA 1: Ejercicio de rimas:**

_Enlace para acceder: https://colab.research.google.com/drive/1phRzdGjHBnLQvY0ZUbRSVYL8eFfn9vgN?usp=sharing_

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

_Enlace para acceder: https://colab.research.google.com/drive/1PYjMSULj92htHh2cnyX_jBl4fihs4BEa?usp=sharing_

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
       
        import polars as pl
        import pandas as pd
        from mlxtend.frequent_patterns import fpgrowth
        from mlxtend.frequent_patterns import association_rules
        
        # Listado de transacciones
        transacciones = [
            ["Milk", "Bread", "Butter"],
            ["Milk", "Bread"],
            ["Bread", "Butter"],
            ["Milk", "Butter"],
            ["Milk", "Bread", "Butter"]
        ]
        
        # Obtener el listado de los artículos únicos
        articulos = sorted(set(item for sublist in transacciones for item in sublist))
        
        # Convertir la lista de transacciones en números binarios
        data_binaria = []
        
        for transaccion in transacciones:
            fila_binaria = [1 if articulo in transaccion else 0 for articulo in articulos]
            data_binaria.append(fila_binaria)
        
        # Crear un binario para DataFrame de Pandas
        df_binario = pd.DataFrame(data_binaria, columns=articulos)
        df_binario = df_binario.astype(bool)  # Convertir a tipo booleano
        
        # Mostrar el DataFrame binario
        print("")
        print("Transacciones en Formato de Númenos Binarios:")
        print("")
        print(df_binario)
        
        # Aplicar FP-Growth para encontrar conjuntos frecuentes
        frequent_itemsets = fpgrowth(df_binario, min_support=0.1, use_colnames=True)
        
        # Mostrar los conjuntos frecuentes en una tabla
        print("")
        print("\nFrequent Itemsets:")
        print("")
        print(frequent_itemsets)
        
        # Generar reglas de asociación a partir de los conjuntos frecuentes
        # Calculamos las reglas con soporte, confianza y lift
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)
        
        # Mostrar las reglas de asociación
        print("")
        print("\n Reglas de Asociación:")
        print("")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        print("")
        print("Desarrollado por: J.E. Carmona-Álvarez")

**3.3. Compare ambos algoritmos con el mismo conjunto de datos**

**TAREA 4: Hopfield y PCA**
https://github.com/GerardoMunoz/ML_2025/blob/main/Hopfield_Covariance.ipynb
**4.1. Buscar el recorrido por todas las ciudades que demore menos tiempo, sin repetir ciudad utilizando redes de Hopfield**

**4.2. Utilizando PCA visualice en 2D una base de datos de MNIST**
