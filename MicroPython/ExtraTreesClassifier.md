**MICROPYTHON** 


**RECURSOS USADOS**

La librería **_math_** proporciona funciones matemáticas comunes para realizar cálculos como trigonométricos, logarítmicos, exponenciales, y de números reales

La librería **_random_** Ofrece una variedad de funciones para generar números aleatorios de diferentes tipos (enteros, reales, extraídos de distribuciones específicas, etc.), así como funciones para seleccionar elementos al azar de listas o combinar el orden de los elementos. 

La librería **_array_**, en el contexto de la programación, sirve para almacenar y organizar datos de forma estructurada y eficiente en memoria. 

    import math
    import random
    import array
    
    class ExtraTreesClassifier:
        def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                     min_samples_leaf=1, max_features='sqrt', random_state=None):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.max_features = max_features
            self.random_state = random_state
            self.estimators = []
            self.classes_ = None
            
            if random_state is not None:
                random.seed(random_state)
        
        def fit(self, X, y):
            # Convertir a arrays si no lo están
            X = self._check_array(X)
            y = self._check_array(y)
            
            # Determinar clases únicas
            self.classes_ = self._unique(y)
            n_classes = len(self.classes_)
            n_samples, n_features = len(X), len(X[0])
            
            # Calcular max_features
            if self.max_features == 'sqrt':
                max_features = int(math.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(math.log2(n_features))
            elif isinstance(self.max_features, (int, float)):
                max_features = self.max_features
            else:
                max_features = n_features
            
            max_features = max(1, min(n_features, max_features))
            
            # Construir árboles
            for _ in range(self.n_estimators):
                tree = self._build_tree(X, y, max_features, n_classes, depth=0)
                self.estimators.append(tree)
        
        def predict(self, X):
            X = self._check_array(X)
            predictions = []
            for sample in X:
                votes = [0] * len(self.classes_)
                for tree in self.estimators:
                    pred = self._predict_tree(sample, tree)
                    votes[self.classes_.index(pred)] += 1
                predictions.append(self.classes_[votes.index(max(votes))])
            return predictions
        
        def _build_tree(self, X, y, max_features, n_classes, depth):
            n_samples = len(X)
            
            # Criterios de parada
            if (self.max_depth is not None and depth >= self.max_depth) or \
               n_samples < self.min_samples_split or \
               self._is_pure(y):
                return {'leaf': True, 'class': self._most_common(y)}
            
            # Seleccionar características aleatorias
            feature_indices = list(range(len(X[0])))
            random.shuffle(feature_indices)
            feature_indices = feature_indices[:max_features]
            
            best_feature, best_threshold, best_gini = None, None, float('inf')
            best_left_indices, best_right_indices = [], []
            
            # Buscar la mejor división
            for feature in feature_indices:
                # Generar umbral aleatorio
                values = [sample[feature] for sample in X]
                min_val, max_val = min(values), max(values)
                threshold = random.uniform(min_val, max_val)
                
                # Dividir los datos
                left_indices = []
                right_indices = []
                for i in range(n_samples):
                    if X[i][feature] <= threshold:
                        left_indices.append(i)
                    else:
                        right_indices.append(i)
                
                # Verificar tamaño mínimo de hojas
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                
                # Calcular impureza Gini
                gini = self._gini_impurity(y, left_indices, right_indices, n_classes)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices
            
            # Si no se encontró una división válida, crear hoja
            if best_feature is None:
                return {'leaf': True, 'class': self._most_common(y)}
            
            # Construir subárboles recursivamente
            left_X = [X[i] for i in best_left_indices]
            left_y = [y[i] for i in best_left_indices]
            right_X = [X[i] for i in best_right_indices]
            right_y = [y[i] for i in best_right_indices]
            
            left_tree = self._build_tree(left_X, left_y, max_features, n_classes, depth+1)
            right_tree = self._build_tree(right_X, right_y, max_features, n_classes, depth+1)
            
            return {
                'leaf': False,
                'feature': best_feature,
                'threshold': best_threshold,
                'left': left_tree,
                'right': right_tree
            }
        
        def _predict_tree(self, sample, tree):
            if tree['leaf']:
                return tree['class']
            
            if sample[tree['feature']] <= tree['threshold']:
                return self._predict_tree(sample, tree['left'])
            else:
                return self._predict_tree(sample, tree['right'])
        
        def _gini_impurity(self, y, left_indices, right_indices, n_classes):
            total = len(left_indices) + len(right_indices)
            if total == 0:
                return 0
            
            # Gini izquierdo
            left_counts = [0] * n_classes
            for i in left_indices:
                left_counts[self.classes_.index(y[i])] += 1
            left_gini = 1.0
            left_size = len(left_indices)
            if left_size > 0:
                for count in left_counts:
                    p = count / left_size
                    left_gini -= p * p
            
            # Gini derecho
            right_counts = [0] * n_classes
            for i in right_indices:
                right_counts[self.classes_.index(y[i])] += 1
            right_gini = 1.0
            right_size = len(right_indices)
            if right_size > 0:
                for count in right_counts:
                    p = count / right_size
                    right_gini -= p * p
            
            # Gini ponderado
            return (left_size * left_gini + right_size * right_gini) / total
        
        def _is_pure(self, y):
            if len(y) == 0:
                return True
            first = y[0]
            for val in y[1:]:
                if val != first:
                    return False
            return True
        
        def _most_common(self, y):
            counts = {}
            for val in y:
                counts[val] = counts.get(val, 0) + 1
            return max(counts.items(), key=lambda x: x[1])[0]
        
        def _unique(self, y):
            seen = []
            for val in y:
                if val not in seen:
                    seen.append(val)
            return seen
        
        def _check_array(self, arr):
            # Convierte la entrada en una lista de listas/valores
            if not isinstance(arr, (list, tuple)):
                raise ValueError("Input must be a list or array")
            if len(arr) == 0:
                return []
            # Si es una lista de números (para y)
            if isinstance(arr[0], (int, float)):
                return list(arr)
            # Si es una lista de listas (para X)
            return [list(row) for row in arr]
