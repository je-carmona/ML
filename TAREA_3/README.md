TAREA 3: Implemente utilizando Polars los siguientes algoritmos para encontrar reglas de asociación:

Enlace Google Colab: https://colab.research.google.com/drive/1PYjMSULj92htHh2cnyX_jBl4fihs4BEa?usp=sharing

3.1. A priori

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
3.2. FP-Crecimiento

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
3.3. Compare ambos algoritmos con el mismo conjunto de datos

Resultados: Reglas de Asociación Apriori: Forma: (9, 5)
![image](https://github.com/user-attachments/assets/3d4c031e-913c-45f5-a0f8-cc670d0051e8)

Resultados: Reglas de asociación FP-Growth: Forma: (12, 5)
![image](https://github.com/user-attachments/assets/6bdd1157-1cfa-4f6b-bf1c-b46a486bc9e1)

