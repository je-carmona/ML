    
    
    
    
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
            self.rng = self.LCG(random_state if random_state is not None else 42)
        
        class LCG:
            """Generador de números aleatorios Linear Congruential"""
            def __init__(self, seed):
                self.state = seed
            
            def next_int(self):
                self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
                return self.state
            
            def random(self):
                return self.next_int() / 4294967296.0
            
            def randrange(self, start, stop):
                return start + int(self.random() * (stop - start))
            
            def shuffle(self, lst):
                """Algoritmo Fisher-Yates shuffle"""
                n = len(lst)
                for i in range(n-1, 0, -1):
                    j = self.randrange(0, i+1)
                    lst[i], lst[j] = lst[j], lst[i]
        
        def fit(self, X, y):
            X = self.list_deep_copy(X)
            y = self.list_copy(y)
            
            # Obtener clases únicas
            self.classes_ = self.get_unique(y)
            n_classes = self.list_length(self.classes_)
            n_samples = self.list_length(X)
            n_features = self.list_length(X[0]) if n_samples > 0 else 0
            
            # Calcular max_features
            if self.max_features == 'sqrt':
                max_features = self.int_sqrt(n_features)
            elif self.max_features == 'log2':
                max_features = self.int_log2(n_features)
            elif isinstance(self.max_features, (int, float)):
                max_features = int(self.max_features)
            else:
                max_features = n_features
            
            max_features = self.max_min(max_features, 1, n_features)
            
            # Construir árboles
            for _ in range(self.n_estimators):
                tree = self.build_tree(X, y, max_features, n_classes, 0)
                self.estimators.append(tree)
        
        def predict(self, X):
            X = self.list_deep_copy(X)
            predictions = []
            n_classes = self.list_length(self.classes_)
            
            for sample in X:
                votes = self.list_init(0, n_classes)
                for tree in self.estimators:
                    pred = self.predict_tree(sample, tree)
                    class_idx = self.find_index(self.classes_, pred)
                    votes[class_idx] += 1
                best_class_idx = self.argmax(votes)
                predictions.append(self.classes_[best_class_idx])
            return predictions
        
        def build_tree(self, X, y, max_features, n_classes, depth):
            n_samples = self.list_length(X)
            
            # Condiciones de parada
            if (self.max_depth is not None and depth >= self.max_depth) or \
               n_samples < self.min_samples_split or \
               self.is_pure(y):
                return {'leaf': True, 'class': self.most_common(y)}
            
            # Seleccionar características aleatorias
            feature_indices = self.list_range(self.list_length(X[0]))
            self.rng.shuffle(feature_indices)
            feature_indices = self.list_slice(feature_indices, 0, max_features)
            
            best_feature, best_threshold = None, None
            best_gini = float('inf')
            best_left_indices, best_right_indices = [], []
            
            for feature in feature_indices:
                # Obtener valores para la característica actual
                values = []
                for sample in X:
                    values.append(sample[feature])
                
                min_val, max_val = self.list_min(values), self.list_max(values)
                threshold = self.rng.random() * (max_val - min_val) + min_val
                
                left_indices, right_indices = [], []
                for i in range(n_samples):
                    if X[i][feature] <= threshold:
                        left_indices.append(i)
                    else:
                        right_indices.append(i)
                
                if (self.list_length(left_indices) < self.min_samples_leaf or 
                    self.list_length(right_indices) < self.min_samples_leaf):
                    continue
                
                gini = self.calculate_gini(y, left_indices, right_indices, n_classes)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices
            
            if best_feature is None:
                return {'leaf': True, 'class': self.most_common(y)}
            
            # Construir subárboles
            left_X, left_y = [], []
            for i in best_left_indices:
                left_X.append(X[i])
                left_y.append(y[i])
            
            right_X, right_y = [], []
            for i in best_right_indices:
                right_X.append(X[i])
                right_y.append(y[i])
            
            left_tree = self.build_tree(left_X, left_y, max_features, n_classes, depth+1)
            right_tree = self.build_tree(right_X, right_y, max_features, n_classes, depth+1)
            
            return {
                'leaf': False,
                'feature': best_feature,
                'threshold': best_threshold,
                'left': left_tree,
                'right': right_tree
            }
        
        def predict_tree(self, sample, tree):
            if tree['leaf']:
                return tree['class']
            
            if sample[tree['feature']] <= tree['threshold']:
                return self.predict_tree(sample, tree['left'])
            else:
                return self.predict_tree(sample, tree['right'])
        
        def calculate_gini(self, y, left_indices, right_indices, n_classes):
            total = self.list_length(left_indices) + self.list_length(right_indices)
            if total == 0:
                return 0.0
            
            # Calcular Gini para nodo izquierdo
            left_counts = self.list_init(0, n_classes)
            for i in left_indices:
                class_idx = self.find_index(self.classes_, y[i])
                left_counts[class_idx] += 1
            
            left_gini = 1.0
            left_size = self.list_length(left_indices)
            if left_size > 0:
                for count in left_counts:
                    prob = count / left_size
                    left_gini -= prob * prob
            
            # Calcular Gini para nodo derecho
            right_counts = self.list_init(0, n_classes)
            for i in right_indices:
                class_idx = self.find_index(self.classes_, y[i])
                right_counts[class_idx] += 1
            
            right_gini = 1.0
            right_size = self.list_length(right_indices)
            if right_size > 0:
                for count in right_counts:
                    prob = count / right_size
                    right_gini -= prob * prob
            
            # Gini ponderado
            return (left_size * left_gini + right_size * right_gini) / total
        
        # ==============================================
        # Funciones auxiliares para manipulación de listas
        # ==============================================
        
        def list_copy(self, lst):
            """Copia superficial de una lista"""
            return [x for x in lst]
        
        def list_deep_copy(self, lst):
            """Copia profunda de una lista de listas"""
            return [self.list_copy(x) if isinstance(x, list) else x for x in lst]
        
        def list_length(self, lst):
            """Longitud de una lista"""
            count = 0
            for _ in lst:
                count += 1
            return count
        
        def list_init(self, value, size):
            """Inicializa una lista con un valor repetido"""
            return [value for _ in range(size)]
        
        def list_range(self, n):
            """Equivalente a range(n) pero devuelve lista"""
            return [i for i in range(n)]
        
        def list_slice(self, lst, start, end):
            """Sub-lista desde start hasta end"""
            return [lst[i] for i in range(start, min(end, self.list_length(lst)))]
        
        def list_min(self, lst):
            """Mínimo valor en una lista"""
            if self.list_length(lst) == 0:
                return float('inf')
            min_val = lst[0]
            for x in lst:
                if x < min_val:
                    min_val = x
            return min_val
        
        def list_max(self, lst):
            """Máximo valor en una lista"""
            if self.list_length(lst) == 0:
                return float('-inf')
            max_val = lst[0]
            for x in lst:
                if x > max_val:
                    max_val = x
            return max_val
        
        def find_index(self, lst, value):
            """Encuentra el índice de un valor en una lista"""
            for i, x in enumerate(lst):
                if x == value:
                    return i
            return -1
        
        def argmax(self, lst):
            """Índice del valor máximo en una lista"""
            max_val = lst[0]
            max_idx = 0
            for i, x in enumerate(lst):
                if x > max_val:
                    max_val = x
                    max_idx = i
            return max_idx
        
        # ==============================================
        # Funciones auxiliares matemáticas
        # ==============================================
        
        def int_sqrt(self, n):
            """Raíz cuadrada entera"""
            if n < 0:
                return 0
            x = n
            y = (x + 1) // 2
            while y < x:
                x = y
                y = (x + n // x) // 2
            return x
        
        def int_log2(self, n):
            """Logaritmo base 2 entero"""
            if n <= 0:
                return 0
            log = 0
            while n > 1:
                n >>= 1
                log += 1
            return log
        
        def max_min(self, val, min_val, max_val):
            """Asegura que val esté entre min_val y max_val"""
            if val < min_val:
                return min_val
            if val > max_val:
                return max_val
            return val
        
        # ==============================================
        # Funciones auxiliares para el árbol
        # ==============================================
        
        def is_pure(self, y):
            """Verifica si todos los valores en y son iguales"""
            if self.list_length(y) == 0:
                return True
            first = y[0]
            for val in y:
                if val != first:
                    return False
            return True
        
        def most_common(self, y):
            """Valor más común en una lista"""
            counts = {}
            for val in y:
                counts[val] = counts.get(val, 0) + 1
            
            max_count = -1
            common_val = None
            for val, count in counts.items():
                if count > max_count:
                    max_count = count
                    common_val = val
            return common_val
        
        def get_unique(self, lst):
            """Lista de valores únicos manteniendo el orden"""
            seen = {}
            result = []
            for x in lst:
                if x not in seen:
                    seen[x] = True
                    result.append(x)
            return result
