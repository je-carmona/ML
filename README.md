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
┌─────────────────────┬─────────────┬───────────┬─────────┬──────────┐
│ Antecedente         ┆ Consecuente ┆ Confianza ┆ Soporte ┆ Lift     │
│ ---                 ┆ ---         ┆ ---       ┆ ---     ┆ ---      │
│ list[str]           ┆ list[str]   ┆ f64       ┆ f64     ┆ f64      │
╞═════════════════════╪═════════════╪═══════════╪═════════╪══════════╡
│ ["Butter"]          ┆ ["Bread"]   ┆ 0.75      ┆ 0.6     ┆ 0.9375   │
│ ["Bread"]           ┆ ["Butter"]  ┆ 0.75      ┆ 0.6     ┆ 0.9375   │
│ ["Milk"]            ┆ ["Bread"]   ┆ 0.75      ┆ 0.6     ┆ 0.9375   │
│ ["Bread"]           ┆ ["Milk"]    ┆ 0.75      ┆ 0.6     ┆ 0.9375   │
│ ["Milk"]            ┆ ["Butter"]  ┆ 0.75      ┆ 0.6     ┆ 0.9375   │
│ ["Butter"]          ┆ ["Milk"]    ┆ 0.75      ┆ 0.6     ┆ 0.9375   │
│ ["Butter", "Milk"]  ┆ ["Bread"]   ┆ 0.666667  ┆ 0.4     ┆ 0.833333 │
│ ["Bread", "Milk"]   ┆ ["Butter"]  ┆ 0.666667  ┆ 0.4     ┆ 0.833333 │
│ ["Bread", "Butter"] ┆ ["Milk"]    ┆ 0.666667  ┆ 0.4     ┆ 0.833333 │
└─────────────────────┴─────────────┴───────────┴─────────┴──────────┘


_Resultados:_ Reglas de asociación  **FP-Growth**:
shape: (12, 5)
╒═════════════════════╤═════════════════════╤══════════════╤═══════════╤══════════╕
│ antecedents         │ consequents         │   confidence │   support │     lift │
╞═════════════════════╪═════════════════════╪══════════════╪═══════════╪══════════╡
│ ['Milk']            │ ['Butter']          │     0.75     │       0.6 │ 0.9375   │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Butter']          │ ['Milk']            │     0.75     │       0.6 │ 0.9375   │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Butter']          │ ['Bread']           │     0.75     │       0.6 │ 0.9375   │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Bread']           │ ['Butter']          │     0.75     │       0.6 │ 0.9375   │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Milk']            │ ['Bread']           │     0.75     │       0.6 │ 0.9375   │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Bread']           │ ['Milk']            │     0.75     │       0.6 │ 0.9375   │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Milk', 'Butter']  │ ['Bread']           │     0.666667 │       0.4 │ 0.833333 │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Milk', 'Bread']   │ ['Butter']          │     0.666667 │       0.4 │ 0.833333 │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Butter', 'Bread'] │ ['Milk']            │     0.666667 │       0.4 │ 0.833333 │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Milk']            │ ['Butter', 'Bread'] │     0.5      │       0.4 │ 0.833333 │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Butter']          │ ['Milk', 'Bread']   │     0.5      │       0.4 │ 0.833333 │
├─────────────────────┼─────────────────────┼──────────────┼───────────┼──────────┤
│ ['Bread']           │ ['Milk', 'Butter']  │     0.5      │       0.4 │ 0.833333 │
╘═════════════════════╧═════════════════════╧══════════════╧═══════════╧══════════╛


**TAREA 4: Hopfield y PCA**
https://github.com/GerardoMunoz/ML_2025/blob/main/Hopfield_Covariance.ipynb
**4.1. Buscar el recorrido por todas las ciudades que demore menos tiempo, sin repetir ciudad utilizando redes de Hopfield**

**4.2. Utilizando PCA visualice en 2D una base de datos de MNIST**
