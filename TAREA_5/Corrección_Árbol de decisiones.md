**CORRECCIÓN - TAREA 5:** Árbol de decisión

_**Enlace Google Colab:**_ [https://colab.research.google.com/drive/17FWE7lUU_2tmAOlBweKxPSowAGShuSRq?usp=sharing](url)

**5.1.** Hacer el algoritmo de árbol de decisión para categorías

        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier, _tree
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        
        # Datos de ejemplo
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
        
        # Preprocesamiento
        X = data[:, :-1]  # Características: Matemáticas, Ciencias, Inglés
        y = data[:, -1]   # Variable objetivo: Carrera
        
        # Codificación
        feature_encoders = [LabelEncoder() for _ in range(X.shape[1])]
        X_encoded = np.column_stack([enc.fit_transform(X[:, i]) for i, enc in enumerate(feature_encoders)])
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)
        
        # Modelo de Árbol de Decisión
        clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, random_state=42)
        clf.fit(X_train, y_train)
        
        # Función para categorizar probabilidades
        def categorize_probability(probs):
            max_prob = max(probs)
            if max_prob < 0.4:
                return "Nada Probable"
            elif max_prob < 0.7:
                return "Poco Probable"
            else:
                return "Muy Probable"
        
        # Visualización del árbol con etiquetas personalizadas
        plt.figure(figsize=(25, 15))
        ax = plt.gca()
        
        # Primero dibujamos el árbol normal
        plot_tree(clf,
                  feature_names=["Matemáticas", "Ciencias", "Inglés"],
                  class_names=label_encoder.classes_,
                  filled=True,
                  rounded=True,
                  ax=ax)
        
        # Obtenemos la estructura del árbol
        tree = clf.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        
        # Creamos una lista para rastrear los nodos hoja
        leaf_nodes = []
        for i in range(n_nodes):
            if children_left[i] == children_right[i]:  # Es un nodo hoja
                leaf_nodes.append(i)
        
        # Asociamos cada texto con su nodo correspondiente
        node_text_mapping = []
        current_leaf_index = 0
        
        for text in ax.texts:
            content = text.get_text()
            if "samples" in content:
                if current_leaf_index < len(leaf_nodes):
                    node_id = leaf_nodes[current_leaf_index]
                    node_text_mapping.append((node_id, text))
                    current_leaf_index += 1
        
        # Actualizamos las etiquetas de los nodos hoja
        for node_id, text in node_text_mapping:
            if children_left[node_id] == children_right[node_id]:  # Es hoja
                values = tree.value[node_id][0]
                total = values.sum()
                probs = values / total
                prob_cat = categorize_probability(probs)
                pred_class = label_encoder.classes_[np.argmax(probs)]
                
                # Construimos la nueva etiqueta
                new_label = f"Nodo {node_id}\n"
                new_label += f"Muestras: {int(total)}\n"
                new_label += f"Clase: {pred_class}\n"
                new_label += f"Confianza: {prob_cat}\n"
                new_label += "Distribución:\n"
                for i, cls in enumerate(label_encoder.classes_):
                    new_label += f"{cls}: {probs[i]:.2f}\n"
                
                text.set_text(new_label)
        
        plt.title("Árbol de Decisión con Categorías de Probabilidad\n(Nada Probable < 40%, Poco Probable 40-70%, Muy Probable > 70%)", pad=20)
        plt.tight_layout()
        plt.show()
        
        # Evaluación del modelo con categorías de probabilidad
        print("\nEvaluación con categorías de probabilidad:")
        y_probs = clf.predict_proba(X_test)
        
        for i in range(len(X_test)):
            original_features = [feature_encoders[j].inverse_transform([X_test[i][j]])[0] for j in range(X_test.shape[1])]
            probs = y_probs[i]
            prob_cat = categorize_probability(probs)
            pred_class = label_encoder.inverse_transform([np.argmax(probs)])[0]
            true_class = label_encoder.inverse_transform([y_test[i]])[0]
            
            print(f"\nEstudiante {i+1}:")
            print(f"Calificaciones: Matemáticas={original_features[0]}, Ciencias={original_features[1]}, Inglés={original_features[2]}")
            print("Probabilidades estimadas:")
            for j, cls in enumerate(label_encoder.classes_):
                print(f"  {cls}: {probs[j]:.2f}")
            print(f"Categoría de confianza: {prob_cat}")
            print(f"Predicción: {pred_class} (Real: {true_class})")
        
        print("\nDesarrollado por: J.E. Carmona-Álvarez")

Resultados:

Evaluación con categorías de probabilidad:

Estudiante 1:
Calificaciones: Matemáticas=G, Ciencias=B, Inglés=G
Probabilidades estimadas:
  A: 1.00
  E: 0.00
  M: 0.00
Categoría de confianza: Muy Probable
Predicción: A (Real: A)

Estudiante 2:
Calificaciones: Matemáticas=R, Ciencias=G, Inglés=B
Probabilidades estimadas:
  A: 0.00
  E: 1.00
  M: 0.00
Categoría de confianza: Muy Probable
Predicción: E (Real: M)

Estudiante 3:
Calificaciones: Matemáticas=G, Ciencias=G, Inglés=G
Probabilidades estimadas:
  A: 0.00
  E: 1.00
  M: 0.00
Categoría de confianza: Muy Probable
Predicción: E (Real: E)

![image](https://github.com/user-attachments/assets/57045df4-d6a1-458c-a488-08bf70da92a7)

Desarrollado por: J.E. Carmona-Álvarez
