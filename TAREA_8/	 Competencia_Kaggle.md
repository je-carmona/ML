**COMPETENCIA DE KAGGEL March Machine Learning Mania 2025**
_**Forecast the 2025 NCAA Basketball Tournaments**_

**Enlace:** https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data

**Enlace Colab:** https://colab.research.google.com/drive/1UGjmn35SlBkCXzpmV4xXo5KZrxpWZLr6?usp=sharing

**NOTA:** Para ejecutar correctamente el código primero debe subir al almacenamiento de Colab el archivo .zip que contiene todas las bases de datos de la competencia. 

**1.** Función para descomprimir el archivo .zip

        !kaggle competitions download -c march-machine-learning-mania-2025
        !unzip march-machine-learning-mania-2025.zip

**2.** _Calculate_team_stats:_ Proceso para calcular los datos estadisticos relevantes de los datos representativos de cada equipo co el fin de determinar cuales son las posibles variables de peso y correlacionables para la predicción. 

        def calculate_team_stats(regular_data):
            # Verificar columnas disponibles
            available_cols = regular_data.columns.tolist()
            print("Columnas disponibles:", available_cols)
        
            # Definir estadísticas básicas que intentaremos calcular
            stats_to_calculate = {
                'W': ['WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
                      'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'],
                'L': ['LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                      'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
            }
        
            # Filtrar solo las columnas que existen en los datos
            w_stats_cols = [col for col in stats_to_calculate['W'] if col in available_cols]
            l_stats_cols = [col for col in stats_to_calculate['L'] if col in available_cols]
        
            # Calcular estadísticas para equipos ganadores
            w_stats = regular_data.groupby(['Season', 'WTeamID'])[w_stats_cols].mean()
            w_stats = w_stats.rename(columns=lambda x: x[1:] if x.startswith('W') else x)
        
            # Calcular estadísticas para equipos perdedores
            l_stats = regular_data.groupby(['Season', 'LTeamID'])[l_stats_cols].mean()
            l_stats = l_stats.rename(columns=lambda x: x[1:] if x.startswith('L') else x)
        
            # Combinar estadísticas
            team_stats = pd.concat([w_stats, l_stats])
            team_stats = team_stats.groupby(level=[0, 1]).mean()  # Promedio si hay duplicados
            team_stats = team_stats.reset_index()
            team_stats = team_stats.rename(columns={'WTeamID': 'TeamID'} if 'WTeamID' in team_stats.columns
                                  else team_stats.rename(columns={'LTeamID': 'TeamID'}))
        
            # Calcular porcentajes solo si tenemos los datos necesarios
            if 'FGM' in team_stats.columns and 'FGA' in team_stats.columns:
                team_stats['FG%'] = team_stats['FGM'] / team_stats['FGA']
            if 'FGM3' in team_stats.columns and 'FGA3' in team_stats.columns:
                team_stats['3P%'] = team_stats['FGM3'] / team_stats['FGA3']
            if 'FTM' in team_stats.columns and 'FTA' in team_stats.columns:
                team_stats['FT%'] = team_stats['FTM'] / team_stats['FTA']
            if 'OR' in team_stats.columns and 'DR' in team_stats.columns:
                team_stats['Rebounds'] = team_stats['OR'] + team_stats['DR']
        
            return team_stats
    
**3.** _Prepare_training_set:_ Presentación y preparación de los datos con los que se va a entrenar el modelo. 

        def prepare_training_set(tourney_data, team_stats):
            if team_stats is None or len(team_stats) == 0:
                raise ValueError("team_stats está vacío o no es válido")
        
            # Verificar columnas mínimas requeridas
            required_cols = ['Season', 'WTeamID', 'LTeamID']
            missing_cols = [col for col in required_cols if col not in tourney_data.columns]
            if missing_cols:
                raise ValueError(f"Faltan columnas requeridas en tourney_data: {missing_cols}")
        
            # Preparar los resultados del torneo
            tourney = tourney_data[required_cols].copy()
        
            # Verificar columnas en team_stats
            if 'TeamID' not in team_stats.columns or 'Season' not in team_stats.columns:
                raise ValueError("team_stats debe contener 'TeamID' y 'Season'")
        
            # Combinar con estadísticas de equipos ganadores
            try:
                tourney = tourney.merge(team_stats,
                                       left_on=['Season', 'WTeamID'],
                                       right_on=['Season', 'TeamID'],
                                       how='left',
                                       suffixes=('', '_W'))
            except Exception as e:
                print(f"Error al fusionar estadísticas ganadoras: {str(e)}")
                return None
        
            # Combinar con estadísticas de equipos perdedores
            try:
                tourney = tourney.merge(team_stats,
                                       left_on=['Season', 'LTeamID'],
                                       right_on=['Season', 'TeamID'],
                                       how='left',
                                       suffixes=('_W', '_L'))
            except Exception as e:
                print(f"Error al fusionar estadísticas perdedoras: {str(e)}")
                return None
        
            # Crear características de diferencia entre equipos
            stat_prefixes = ['Score', 'FG', '3P', 'FT', 'Rebounds', 'Ast', 'TO', 'Stl', 'Blk']
            for stat in stat_prefixes:
                w_col = f"{stat}_W" if f"{stat}_W" in tourney.columns else stat
                l_col = f"{stat}_L" if f"{stat}_L" in tourney.columns else stat
        
                if w_col in tourney.columns and l_col in tourney.columns:
                    tourney[f'{stat}_Diff'] = tourney[w_col] - tourney[l_col]
                else:
                    print(f"Advertencia: No se encontraron columnas para {stat}")
        
            # La variable objetivo es 1 si el primer equipo gana
            tourney['target'] = 1
        
            # También necesitamos ejemplos donde el orden de los equipos se invierte
            reverse = tourney.copy()
            for col in tourney.columns:
                if col.endswith('_Diff'):
                    reverse[col] = -reverse[col]
            reverse['target'] = 0
        
            # Combinar ambos conjuntos
            full_data = pd.concat([tourney, reverse])
        
            return full_data

**3.1.** Verificacion de las columnas reales en los datos:

        import os

        # Listar archivos en el directorio actual
        print("Archivos en el directorio actual:", os.listdir())

---

        import pandas as pd
        from IPython.display import display
        
        def load_and_display_data(gender):
            """Carga y muestra los datos para un género específico"""
            prefix = gender
            gender_name = 'Masculino' if gender == 'M' else 'Femenino'
        
            print(f"\n=== DATOS {gender_name.upper()} ===")
        
            try:
                # Cargar archivos
                regular = pd.read_csv(f'{prefix}RegularSeasonDetailedResults.csv')
                tourney = pd.read_csv(f'{prefix}NCAATourneyCompactResults.csv')
                seeds = pd.read_csv(f'{prefix}NCAATourneySeeds.csv')
        
                # Mostrar información básica
                print(f"\n1. Temporada Regular ({len(regular)} registros):")
                display(regular.head(3))
                print("\nColumnas disponibles:", regular.columns.tolist())
        
                print(f"\n2. Torneo NCAA ({len(tourney)} registros):")
                display(tourney.head(3))
                print("\nColumnas disponibles:", tourney.columns.tolist())
        
                print(f"\n3. Semillas del Torneo ({len(seeds)} registros):")
                display(seeds.head(3))
                print("\nColumnas disponibles:", seeds.columns.tolist())
        
                return regular, tourney, seeds
        
            except Exception as e:
                print(f"Error al cargar datos {gender_name.lower()}: {str(e)}")
                return None, None, None
        
        # Mostrar datos masculinos
        m_regular, m_tourney, m_seeds = load_and_display_data('M')
        
        # Mostrar datos femeninos
        w_regular, w_tourney, w_seeds = load_and_display_data('W')
        
        # Comparativa básica
        if m_regular is not None and w_regular is not None:
            print("\n=== COMPARATIVA ===")
            print(f"Partidos temporada regular: Masculino={len(m_regular)}, Femenino={len(w_regular)}")
            print(f"Partidos de torneo: Masculino={len(m_tourney)}, Femenino={len(w_tourney)}")
            print(f"Equipos con semilla: Masculino={len(m_seeds)}, Femenino={len(w_seeds)}")

**RESULTADOS**

_**=== DATOS MASCULINO ===**_

1. Temporada Regular (118882 registros):

![image](https://github.com/user-attachments/assets/8bc673a0-0dcc-4d6c-932e-1cd0eae2c93a)

Columnas disponibles: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

2. Torneo NCAA (2518 registros):

![image](https://github.com/user-attachments/assets/92053600-dcd4-4230-9ad5-77a8bcf28c94)

Columnas disponibles: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

3. Semillas del Torneo (2626 registros):

![image](https://github.com/user-attachments/assets/9b8e6c2d-2dfb-4866-83dc-bfaf33a62f3c)

Columnas disponibles: ['Season', 'Seed', 'TeamID']

**_=== DATOS FEMENINO ===_**

1. Temporada Regular (81708 registros):

![image](https://github.com/user-attachments/assets/6424a9b6-82ad-4fcd-a2b1-fe5c2403330a)

Columnas disponibles: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

2. Torneo NCAA (1650 registros):

![image](https://github.com/user-attachments/assets/b4dfeaa9-0dca-40b1-af92-760ac173a7e5)

Columnas disponibles: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

3. Semillas del Torneo (1744 registros):

![image](https://github.com/user-attachments/assets/23a41c81-437f-4a41-90aa-cd751a28c0c0)

Columnas disponibles: ['Season', 'Seed', 'TeamID']

_**=== COMPARATIVA ===**_
Partidos temporada regular: Masculino=118882, Femenino=81708
Partidos de torneo: Masculino=2518, Femenino=1650
Equipos con semilla: Masculino=2626, Femenino=1744

**3.2.** Correlación y normalización de datos para el entrenamiento:

        import pandas as pd
        from IPython.display import display, HTML
        
        def display_data_with_info(df, title):
            """Muestra un DataFrame con información detallada"""
            display(HTML(f"<h3>{title}</h3>"))
            display(df.head(3))
            print(f"\nFilas: {len(df)} | Columnas: {len(df.columns)}")
            print("Columnas:", df.columns.tolist())
            print("\nEstadísticas descriptivas:")
            display(df.describe(include='all').head(3))
        
        def analyze_gender_data(gender):
            """Analiza y muestra datos para un género"""
            prefix = gender
            gender_name = 'Masculino' if gender == 'M' else 'Femenino'
        
            print(f"\n{'='*50}")
            print(f"{' ANÁLISIS DE DATOS ' + gender_name.upper() + ' ':=^50}")
            print(f"{'='*50}")
        
            try:
                # Cargar datos
                regular = pd.read_csv(f'{prefix}RegularSeasonDetailedResults.csv')
                tourney = pd.read_csv(f'{prefix}NCAATourneyCompactResults.csv')
                seeds = pd.read_csv(f'{prefix}NCAATourneySeeds.csv')
                teams = pd.read_csv(f'{prefix}Teams.csv')
        
                # Mostrar datos
                display_data_with_info(regular, f"1. Temporada Regular {gender_name}")
                display_data_with_info(tourney, f"2. Torneo NCAA {gender_name}")
                display_data_with_info(seeds, f"3. Semillas del Torneo {gender_name}")
                display_data_with_info(teams, f"4. Equipos {gender_name}")
        
                return regular, tourney, seeds, teams
        
            except Exception as e:
                print(f"Error al cargar datos {gender_name.lower()}: {str(e)}")
                return None, None, None, None
        
        # Analizar ambos géneros
        m_data = analyze_gender_data('M')
        w_data = analyze_gender_data('W')

**_RESULTADOS:_**

**_ANÁLISIS DE DATOS MASCULINO_**

1. Temporada Regular Masculino

![image](https://github.com/user-attachments/assets/e9ef7d25-7c4e-4cec-94b9-6556816f7ed7)

Filas: 118882 | Columnas: 34
Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/44a430c7-a089-4b0c-b216-852063af52f4)

2. Torneo NCAA Masculino

![image](https://github.com/user-attachments/assets/a2372818-d538-46b5-8c53-d6da860f81d4)

Filas: 2518 | Columnas: 8
Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/af7da940-159d-4d1c-98a9-4dc2805756e5)

3. Semillas del Torneo Masculino

![image](https://github.com/user-attachments/assets/d6a42604-bfd3-4ddd-aaf5-39320e4cc2d8)

Filas: 2626 | Columnas: 3
Columnas: ['Season', 'Seed', 'TeamID']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/5f6d31cc-1b99-4048-acbd-7a465bae3f3f)

4. Equipos Masculino

![image](https://github.com/user-attachments/assets/a754bf16-42a3-47f1-9e9a-aea122c454b0)

Filas: 380 | Columnas: 4
Columnas: ['TeamID', 'TeamName', 'FirstD1Season', 'LastD1Season']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/a87de114-9b12-42b6-a4ab-a30b2b483f74)

_**ANÁLISIS DE DATOS FEMENINO**_

1. Temporada Regular Femenino

![image](https://github.com/user-attachments/assets/49a2342e-d40a-4429-b217-6f567a140bc6)

Filas: 81708 | Columnas: 34
Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

2. Torneo NCAA Femenino

![image](https://github.com/user-attachments/assets/d645af36-60f2-4bd4-8d1c-70e930ba91b1)

Filas: 1650 | Columnas: 8
Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/ad825fee-5e30-4578-af6e-4e9950441250)

3. Semillas del Torneo Femenino

![image](https://github.com/user-attachments/assets/cf9611f0-4ab6-4eb2-83cc-c35aa0d3e6e7)

Filas: 1744 | Columnas: 3
Columnas: ['Season', 'Seed', 'TeamID']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/2fd413d6-7e65-4630-965a-a0291c4bb89d)

4. Equipos Femenino

![image](https://github.com/user-attachments/assets/91a2790a-a223-4dad-a9aa-de327f115f09)


Filas: 378 | Columnas: 2
Columnas: ['TeamID', 'TeamName']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/07377c8f-8a35-4df1-801b-5754a38f997e)

**4.** Cálculo de las estadísticascon los datos relevantes del entrenamiento del modelo:

        def calculate_team_stats(regular_data):
            # Verificar que tenemos las columnas mínimas necesarias
            required_cols = ['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']
            missing_cols = [col for col in required_cols if col not in regular_data.columns]
            if missing_cols:
                raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
        
            # Estadísticas básicas para equipos ganadores
            w_stats = regular_data.groupby(['Season', 'WTeamID']).agg({
                'WScore': 'mean',
                'LScore': 'mean',
                'WFGM': 'mean',
                'WFGA': 'mean',
                'WFGM3': 'mean',
                'WFGA3': 'mean',
                'WFTM': 'mean',
                'WFTA': 'mean',
                'WOR': 'mean',
                'WDR': 'mean',
                'WAst': 'mean',
                'WTO': 'mean',
                'WStl': 'mean',
                'WBlk': 'mean',
                'WPF': 'mean'
            }).reset_index()
        
            # Renombrar columnas (quitamos la W inicial)
            w_stats = w_stats.rename(columns={
                'WTeamID': 'TeamID',
                **{col: col[1:] for col in w_stats.columns if col.startswith('W') and col not in ['WTeamID']}
            })
        
            # Estadísticas básicas para equipos perdedores
            l_stats = regular_data.groupby(['Season', 'LTeamID']).agg({
                'LScore': 'mean',
                'WScore': 'mean',
                'LFGM': 'mean',
                'LFGA': 'mean',
                'LFGM3': 'mean',
                'LFGA3': 'mean',
                'LFTM': 'mean',
                'LFTA': 'mean',
                'LOR': 'mean',
                'LDR': 'mean',
                'LAst': 'mean',
                'LTO': 'mean',
                'LStl': 'mean',
                'LBlk': 'mean',
                'LPF': 'mean'
            }).reset_index()
        
            # Renombrar columnas (quitamos la L inicial)
            l_stats = l_stats.rename(columns={
                'LTeamID': 'TeamID',
                **{col: col[1:] for col in l_stats.columns if col.startswith('L') and col not in ['LTeamID']}
            })
        
            # Combinar estadísticas (promedio de ambas perspectivas)
            team_stats = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).mean().reset_index()
        
            # Calcular métricas derivadas
            if 'FGM' in team_stats.columns and 'FGA' in team_stats.columns:
                team_stats['FG%'] = team_stats['FGM'] / team_stats['FGA']
            if 'FGM3' in team_stats.columns and 'FGA3' in team_stats.columns:
                team_stats['3P%'] = team_stats['FGM3'] / team_stats['FGA3']
            if 'FTM' in team_stats.columns and 'FTA' in team_stats.columns:
                team_stats['FT%'] = team_stats['FTM'] / team_stats['FTA']
            if 'OR' in team_stats.columns and 'DR' in team_stats.columns:
                team_stats['Rebounds'] = team_stats['OR'] + team_stats['DR']
            if 'Ast' in team_stats.columns and 'TO' in team_stats.columns:
                team_stats['Ast/TO'] = team_stats['Ast'] / team_stats['TO']
        
            return team_stats

**5.** Entrenamiento y resultados del modelo: 

        import pandas as pd
        import numpy as np
        import os
        
        def load_and_validate_data(gender):
            """Carga y valida los datos con manejo de errores mejorado"""
            try:
                prefix = gender
                print(f"\nCargando datos para el torneo {'masculino' if gender == 'M' else 'femenino'}...")
        
                # Cargar archivos con verificación
                teams = pd.read_csv(f'{prefix}Teams.csv')
                seeds = pd.read_csv(f'{prefix}NCAATourneySeeds.csv')
                regular = pd.read_csv(f'{prefix}RegularSeasonCompactResults.csv')
                tourney = pd.read_csv(f'{prefix}NCAATourneyCompactResults.csv')
        
                return teams, seeds, regular, tourney
        
            except Exception as e:
                print(f"Error cargando datos: {e}")
                return None, None, None, None
        
        def build_simple_model(regular_data):
            """Construye un modelo simple basado en semillas y puntuaciones"""
            try:
                # Calcular estadísticas básicas
                win_stats = regular_data.groupby('WTeamID').agg({
                    'WScore': ['mean', 'max', 'min'],
                    'LTeamID': 'count'
                }).reset_index()
                win_stats.columns = ['TeamID', 'OffenseAvg', 'OffenseMax', 'OffenseMin', 'Wins']
        
                loss_stats = regular_data.groupby('LTeamID').agg({
                    'LScore': ['mean', 'max', 'min'],
                    'WTeamID': 'count'
                }).reset_index()
                loss_stats.columns = ['TeamID', 'DefenseAvg', 'DefenseMax', 'DefenseMin', 'Losses']
        
                team_stats = pd.merge(
                    win_stats,
                    loss_stats,
                    on='TeamID',
                    how='outer'
                ).fillna(0)
        
                # Calcular porcentaje de victorias
                team_stats['WinPct'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])
                team_stats['Strength'] = team_stats['OffenseAvg'] * 0.7 + team_stats['DefenseAvg'] * 0.3
        
                return team_stats
        
            except Exception as e:
                print(f"Error construyendo modelo simple: {e}")
                return None
        
        def predict_winners(team_stats, seeds, teams, current_season=2025):
            """Predice ganadores usando un enfoque basado en semillas y estadísticas"""
            try:
                # Procesar semillas
                seeds = seeds[seeds['Season'] == current_season].copy()
                seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)
        
                # Combinar con estadísticas y datos de equipos
                predictions = seeds.merge(
                    team_stats,
                    on='TeamID',
                    how='left'
                ).merge(
                    teams[['TeamID', 'TeamName']],
                    on='TeamID',
                    how='left'
                )
        
                # Si faltan estadísticas, usar valores basados en semilla
                predictions['OffenseAvg'] = predictions['OffenseAvg'].fillna(80 - predictions['SeedNum'])
                predictions['DefenseAvg'] = predictions['DefenseAvg'].fillna(60 + predictions['SeedNum'])
                predictions['Strength'] = predictions['Strength'].fillna(
                    predictions['OffenseAvg'] * 0.7 + predictions['DefenseAvg'] * 0.3
                )
        
                return predictions
        
            except Exception as e:
                print(f"Error generando predicciones: {e}")
                return None
        
        def simulate_round(teams_df, round_name):
            """Simula una ronda del torneo y devuelve resultados detallados"""
            try:
                # Ordenar por semilla
                sorted_teams = teams_df.sort_values('SeedNum')
                winners = []
                detailed_results = []
        
                # Determinar emparejamientos según la ronda
                if round_name == "Ronda 1":
                    top = sorted_teams.iloc[:8]
                    bottom = sorted_teams.iloc[8:16].iloc[::-1]
                    matchups = list(zip(top.iterrows(), bottom.iterrows()))
                elif round_name == "Ronda 2":
                    if len(sorted_teams) == 8:
                        matchups = list(zip(sorted_teams.iloc[::2].iterrows(), sorted_teams.iloc[1::2].iterrows()))
                    else:
                        matchups = []
                elif round_name == "Sweet 16":
                    if len(sorted_teams) == 4:
                        matchups = list(zip(sorted_teams.iloc[::2].iterrows(), sorted_teams.iloc[1::2].iterrows()))
                    else:
                        matchups = []
                elif round_name == "Elite 8":
                    if len(sorted_teams) == 2:
                        matchups = [(sorted_teams.iloc[0:1].iterrows().__next__(), sorted_teams.iloc[1:2].iterrows().__next__())]
                    else:
                        matchups = []
                else:
                    matchups = []
        
                for (_, team1), (_, team2) in matchups:
                    # Calcular probabilidad basada en estadísticas
                    prob = 1 / (1 + np.exp(-(team1['Strength'] - team2['Strength'])/10))
                    
                    if prob >= 0.5:
                        winner = team1
                        winner_prob = prob
                    else:
                        winner = team2
                        winner_prob = 1 - prob
        
                    # Resultados detallados para CSV
                    detailed_results.append({
                        'Round': round_name,
                        'Team1_ID': team1['TeamID'],
                        'Team1_Name': team1['TeamName'],
                        'Team1_Seed': team1['Seed'],
                        'Team1_OffenseAvg': team1['OffenseAvg'],
                        'Team1_DefenseAvg': team1['DefenseAvg'],
                        'Team1_Strength': team1['Strength'],
                        'Team2_ID': team2['TeamID'],
                        'Team2_Name': team2['TeamName'],
                        'Team2_Seed': team2['Seed'],
                        'Team2_OffenseAvg': team2['OffenseAvg'],
                        'Team2_DefenseAvg': team2['DefenseAvg'],
                        'Team2_Strength': team2['Strength'],
                        'Probability': winner_prob,
                        'Predicted_Winner_ID': winner['TeamID'],
                        'Predicted_Winner_Name': winner['TeamName']
                    })
        
                    winners.append(winner)
        
                # Crear DataFrames
                detailed_df = pd.DataFrame(detailed_results)
                
                if winners:
                    winners_df = pd.DataFrame(winners)
                else:
                    winners_df = pd.DataFrame(columns=teams_df.columns)
        
                return winners_df, detailed_df
        
            except Exception as e:
                print(f"Error en simulate_round: {e}")
                return pd.DataFrame(columns=teams_df.columns), pd.DataFrame()
        
        def save_combined_predictions(all_predictions, filename='submission.csv'):
            """Guarda todas las predicciones en un solo archivo CSV"""
            try:
                # Combinar todas las predicciones
                combined_df = pd.concat(all_predictions, ignore_index=True)
                
                # Ordenar por género y ronda
                combined_df = combined_df.sort_values(['Gender', 'Round'])
                
                # Guardar CSV
                combined_df.to_csv(filename, index=False)
                print(f"\nPredicciones combinadas guardadas en {filename}")
                
                return filename
                
            except Exception as e:
                print(f"Error guardando predicciones combinadas: {e}")
                return None
        
        def predict_tournament(gender):
            """Predice el torneo completo para un género"""
            try:
                # Cargar datos
                teams, seeds, regular, tourney = load_and_validate_data(gender)
                if teams is None:
                    return None
        
                # Construir modelo simple
                team_stats = build_simple_model(regular)
                if team_stats is None:
                    return None
        
                # Predecir ganadores
                predictions = predict_winners(team_stats, seeds, teams)
                if predictions is None:
                    return None
        
                # Simular rondas
                round_names = ["Ronda 1", "Ronda 2", "Sweet 16", "Elite 8", "Final"]
                all_detailed_results = []
                current_teams = predictions.copy()
        
                print(f"\n{'*'*50}")
                print(f"PREDICCIONES TORNEO {'MASCULINO' if gender == 'M' else 'FEMENINO'} 2025")
                print(f"{'*'*50}")
        
                for round_name in round_names:
                    if len(current_teams) < 2:
                        print(f"\nNo hay suficientes equipos para continuar ({len(current_teams)} restantes)")
                        break
        
                    winners, detailed = simulate_round(current_teams, round_name)
                    
                    if not detailed.empty:
                        all_detailed_results.append(detailed)
                    
                    current_teams = winners
        
                # Añadir género a los resultados
                if all_detailed_results:
                    detailed_df = pd.concat(all_detailed_results, ignore_index=True)
                    detailed_df['Gender'] = 'M' if gender == 'M' else 'F'
                    return detailed_df
                else:
                    print("\nNo se generaron predicciones para este torneo.")
                    return None
        
            except Exception as e:
                print(f"Error en predict_tournament: {e}")
                return None
        
        # Ejecutar predicciones y guardar resultados
        print("PREDICCIÓN COMBINADA DE GANADORES - NCAA 2025")
        
        # Predecir ambos torneos y combinar resultados
        male_predictions = predict_tournament('M')
        female_predictions = predict_tournament('W')
        
        # Combinar todos los resultados en un solo DataFrame
        all_predictions = []
        if male_predictions is not None:
            all_predictions.append(male_predictions)
        if female_predictions is not None:
            all_predictions.append(female_predictions)
        
        if all_predictions:
            # Guardar archivo combinado
            combined_file = save_combined_predictions(all_predictions)
            
            # Mostrar resumen
            print("\nResumen de predicciones combinadas:")
            combined_df = pd.concat(all_predictions, ignore_index=True)
            print(combined_df[['Gender', 'Round', 'Team1_Name', 'Team2_Name', 'Probability', 'Predicted_Winner_Name']].to_string(index=False))
            
            if combined_file:
                print(f"\nArchivo combinado generado con éxito: {combined_file}")
        else:
            print("\nNo se generaron predicciones para guardar.")
        
        print("\nProceso completado.")
        
        print("\nDesarrollado por: J.E. Carmona Alvarez & J. Ortiz-Aguilar")

**_RESULTADOS:_**

PREDICCIÓN COMBINADA DE GANADORES - NCAA 2025

Cargando datos para el torneo masculino...

**************************************************
PREDICCIONES TORNEO MASCULINO 2025
**************************************************

No hay suficientes equipos para continuar (1 restantes)

Cargando datos para el torneo femenino...

**************************************************
PREDICCIONES TORNEO FEMENINO 2025
**************************************************

No hay suficientes equipos para continuar (1 restantes)

Predicciones combinadas guardadas en submission.csv

Resumen de predicciones combinadas:
Gender    Round     Team1_Name     Team2_Name  Probability Predicted_Winner_Name
     M  Ronda 1           Duke        Arizona     0.552244                  Duke
     M  Ronda 1        Houston         Purdue     0.520029               Houston
     M  Ronda 1        Florida       Maryland     0.530321              Maryland
     M  Ronda 1         Auburn      Texas A&M     0.594417                Auburn
     M  Ronda 1    Michigan St        Iowa St     0.565454               Iowa St
     M  Ronda 1      St John's     Texas Tech     0.527960            Texas Tech
     M  Ronda 1      Tennessee       Kentucky     0.558968              Kentucky
     M  Ronda 1        Alabama      Wisconsin     0.623066               Alabama
     M  Ronda 2           Duke        Houston     0.648415                  Duke
     M  Ronda 2         Auburn        Alabama     0.538614                Auburn
     M  Ronda 2     Texas Tech        Iowa St     0.538060               Iowa St
     M  Ronda 2       Kentucky       Maryland     0.507902              Kentucky
     M Sweet 16           Duke         Auburn     0.621982                  Duke
     M Sweet 16        Iowa St       Kentucky     0.531995              Kentucky
     M  Elite 8           Duke       Kentucky     0.587594                  Duke
     F  Ronda 1 South Carolina       Maryland     0.614750              Maryland
     F  Ronda 1          Texas        Ohio St     0.544421               Ohio St
     F  Ronda 1            USC       Kentucky     0.558225              Kentucky
     F  Ronda 1           UCLA         Baylor     0.591130                Baylor
     F  Ronda 1       NC State            LSU     0.509647              NC State
     F  Ronda 1    Connecticut       Oklahoma     0.544057           Connecticut
     F  Ronda 1            TCU     Notre Dame     0.585883            Notre Dame
     F  Ronda 1           Duke North Carolina     0.587293        North Carolina
     F  Ronda 2    Connecticut       NC State     0.677322           Connecticut
     F  Ronda 2 North Carolina     Notre Dame     0.538414        North Carolina
     F  Ronda 2         Baylor       Kentucky     0.592432                Baylor
     F  Ronda 2        Ohio St       Maryland     0.549821              Maryland
     F Sweet 16    Connecticut North Carolina     0.540603           Connecticut
     F Sweet 16         Baylor       Maryland     0.512536              Maryland
     F  Elite 8    Connecticut       Maryland     0.542977           Connecticut

Archivo combinado generado con éxito: submission.csv

Proceso completado.

Desarrollado por: J.E. Carmona Alvarez & J. Ortiz-Aguilar

