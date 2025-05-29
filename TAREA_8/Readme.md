**COMPETENCIA DE KAGGEL March Machine Learning Mania 2025**
_**Forecast the 2025 NCAA Basketball Tournaments**_

**Enlace:** https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data

**Enlace Colab:** https://colab.research.google.com/drive/1UGjmn35SlBkCXzpmV4xXo5KZrxpWZLr6?usp=sharing

1. Desarrollo

        !kaggle competitions download -c march-machine-learning-mania-2025
        !unzip march-machine-learning-mania-2025.zip
        
        #verifiquemos las columnas reales en tus datos:
        
        def load_data(gender='M', data_path=''):
            prefix = gender if gender in ['M', 'W'] else 'M'
            try:
                regular = pd.read_csv(f'{data_path}{prefix}RegularSeasonDetailedResults.csv')
                print("Columnas en RegularSeasonDetailedResults:", regular.columns.tolist())
                tourney = pd.read_csv(f'{data_path}{prefix}NCAATourneyCompactResults.csv')
                print("\nColumnas en NCAATourneyCompactResults:", tourney.columns.tolist())
                seeds = pd.read_csv(f'{data_path}{prefix}NCAATourneySeeds.csv')
                return regular, tourney, seeds
            except Exception as e:
                print(f"Error al cargar datos: {str(e)}")
                return None, None, None
        
        # calculate_team_stats
        
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
        
        # prepare_training_set
        
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
        
        # ----- PRESENTACIÓN DE DATOS 
        
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
        
        # calcula estadísticas
        
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
        
        # -------- PREDICCIÓN 
        
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        from itertools import combinations
        
        def load_and_validate_data(gender):
            """Carga y valida los datos con manejo de errores mejorado"""
            try:
                prefix = gender
                print(f"\nCargando datos para el torneo {'masculino' if gender == 'M' else 'femenino'}...")
                
                # Cargar archivos con verificación
                teams = pd.read_csv(f'{prefix}Teams.csv')
                seeds = pd.read_csv(f'{prefix}NCAATourneySeeds.csv')
                regular = pd.read_csv(f'{prefix}RegularSeasonCompactResults.csv')  # Usamos versión compacta
                tourney = pd.read_csv(f'{prefix}NCAATourneyCompactResults.csv')
                
                # Verificar columnas disponibles
                print("\nColumnas en cada archivo:")
                print(f"Teams: {teams.columns.tolist()}")
                print(f"Seeds: {seeds.columns.tolist()}")
                print(f"Regular: {regular.columns.tolist()}")
                print(f"Tourney: {tourney.columns.tolist()}")
                
                # Renombrar columnas si es necesario (para compatibilidad)
                if 'WTeamID' not in regular.columns:
                    if 'Winner' in regular.columns:  # Ejemplo de adaptación
                        regular = regular.rename(columns={'Winner': 'WTeamID', 'Loser': 'LTeamID'})
                    else:
                        raise ValueError("No se encontraron columnas de equipos ganadores/perdedores")
                
                return teams, seeds, regular, tourney
            
            except Exception as e:
                print(f"Error cargando datos: {e}")
                return None, None, None, None
        
        def build_simple_model(regular_data):
            """Construye un modelo simple basado en semillas y puntuaciones"""
            try:
                # Calcular estadísticas básicas
                win_stats = regular_data.groupby('WTeamID')['WScore'].mean().reset_index()
                loss_stats = regular_data.groupby('LTeamID')['LScore'].mean().reset_index()
                
                team_stats = pd.merge(
                    win_stats.rename(columns={'WTeamID': 'TeamID', 'WScore': 'Offense'}),
                    loss_stats.rename(columns={'LTeamID': 'TeamID', 'LScore': 'Defense'}),
                    on='TeamID',
                    how='outer'
                ).fillna(0)
                
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
                
                # Combinar con estadísticas
                predictions = seeds.merge(
                    team_stats,
                    on='TeamID',
                    how='left'
                ).merge(
                    teams[['TeamID', 'TeamName']],
                    on='TeamID',
                    how='left'
                )
                
                # Si faltan estadísticas, usar solo la semilla
                predictions['Offense'] = predictions['Offense'].fillna(80 - predictions['SeedNum'])
                predictions['Defense'] = predictions['Defense'].fillna(60 + predictions['SeedNum'])
                
                return predictions[['TeamID', 'TeamName', 'Seed', 'SeedNum', 'Offense', 'Defense']]
            
            except Exception as e:
                print(f"Error generando predicciones: {e}")
                return None
        
        def simulate_round(teams_df, round_name):
            """Simula una ronda del torneo"""
            # Ordenar por semilla
            sorted_teams = teams_df.sort_values('SeedNum')
            
            # Emparejar equipos según la ronda
            if round_name == "Ronda 1":
                matchups = list(zip(sorted_teams.iloc[::2].values, sorted_teams.iloc[1::2].values))
            elif round_name == "Ronda 2":
                matchups = list(zip(sorted_teams.iloc[:8].values, sorted_teams.iloc[8:16].values))
            else:
                matchups = []
            
            winners = []
            results = []
            
            for team1, team2 in matchups:
                # Calcular probabilidad simple basada en semillas y estadísticas
                prob = 0.5 + (team2[4] - team1[4])/100  # Ajuste basado en ofensa
                
                if prob >= 0.5:
                    winner = team1
                    prob = min(0.95, prob)  # Limitar probabilidad máxima
                else:
                    winner = team2
                    prob = max(0.05, 1-prob)
                
                results.append({
                    'Ronda': round_name,
                    'Equipo1': team1[1],  # TeamName
                    'Equipo2': team2[1],
                    'Probabilidad': f"{prob:.2f}",
                    'Ganador': winner[1]
                })
                winners.append(winner)
            
            return pd.DataFrame(results), pd.DataFrame(winners, columns=teams_df.columns)
        
        def display_round_results(results_df):
            """Muestra los resultados de una ronda"""
            print(f"\n{results_df['Ronda'].iloc[0].upper()}:")
            for _, row in results_df.iterrows():
                print(f"  {row['Equipo1']} vs {row['Equipo2']}")
                print(f"  Probabilidad: {row['Probabilidad']} -> GANADOR: {row['Ganador']}")
                print("  " + "-"*40)
        
        def predict_tournament(gender):
            """Predice el torneo completo para un género"""
            # Cargar datos
            teams, seeds, regular, tourney = load_and_validate_data(gender)
            if teams is None:
                return
            
            # Construir modelo simple
            team_stats = build_simple_model(regular)
            if team_stats is None:
                return
            
            # Predecir ganadores
            predictions = predict_winners(team_stats, seeds, teams)
            if predictions is None:
                return
            
            # Simular rondas
            round_names = ["Ronda 1", "Ronda 2", "Sweet 16", "Elite 8", "Final Four", "Final"]
            
            print(f"\n{'*'*50}")
            print(f"PREDICCIONES TORNEO {'MASCULINO' if gender == 'M' else 'FEMENINO'} 2025")
            print(f"{'*'*50}")
            
            current_teams = predictions.copy()
            
            for round_name in round_names:
                if len(current_teams) < 2:
                    break
                    
                results, winners = simulate_round(current_teams, round_name)
                display_round_results(results)
                current_teams = winners
        
        # Ejecutar predicciones
        print("PREDICCIÓN DE GANADORES POR RONDA - NCAA 2025")
        predict_tournament('M')  # Torneo masculino
        predict_tournament('W')  # Torneo femenino
        
        
        print("Desarrollado por: J.E. Carmona Alvarez & J. Ortiz- Aguilar")


2. Resultados

Traceback (most recent call last):
  File "/usr/local/bin/kaggle", line 4, in <module>
    from kaggle.cli import main
  File "/usr/local/lib/python3.11/dist-packages/kaggle/__init__.py", line 6, in <module>
    api.authenticate()
  File "/usr/local/lib/python3.11/dist-packages/kaggle/api/kaggle_api_extended.py", line 434, in authenticate
    raise IOError('Could not find {}. Make sure it\'s located in'
OSError: Could not find kaggle.json. Make sure it's located in /root/.config/kaggle. Or use the environment method. See setup instructions at https://github.com/Kaggle/kaggle-api/
Archive:  march-machine-learning-mania-2025.zip
replace Cities.csv? [y]es, [n]o, [A]ll, [N]one, [r]ename: A
  inflating: Cities.csv              
  inflating: Conferences.csv         
  inflating: MConferenceTourneyGames.csv  
  inflating: MGameCities.csv         
  inflating: MMasseyOrdinals.csv     
  inflating: MNCAATourneyCompactResults.csv  
  inflating: MNCAATourneyDetailedResults.csv  
  inflating: MNCAATourneySeedRoundSlots.csv  
  inflating: MNCAATourneySeeds.csv   
  inflating: MNCAATourneySlots.csv   
  inflating: MRegularSeasonCompactResults.csv  
  inflating: MRegularSeasonDetailedResults.csv  
  inflating: MSeasons.csv            
  inflating: MSecondaryTourneyCompactResults.csv  
  inflating: MSecondaryTourneyTeams.csv  
  inflating: MTeamCoaches.csv        
  inflating: MTeamConferences.csv    
  inflating: MTeamSpellings.csv      
  inflating: MTeams.csv              
  inflating: SampleSubmissionStage1.csv  
  inflating: SampleSubmissionStage2.csv  
  inflating: SeedBenchmarkStage1.csv  
  inflating: WConferenceTourneyGames.csv  
  inflating: WGameCities.csv         
  inflating: WNCAATourneyCompactResults.csv  
  inflating: WNCAATourneyDetailedResults.csv  
  inflating: WNCAATourneySeeds.csv   
  inflating: WNCAATourneySlots.csv   
  inflating: WRegularSeasonCompactResults.csv  
  inflating: WRegularSeasonDetailedResults.csv  
  inflating: WSeasons.csv            
  inflating: WSecondaryTourneyCompactResults.csv  
  inflating: WSecondaryTourneyTeams.csv  
  inflating: WTeamConferences.csv    
  inflating: WTeamSpellings.csv      
  inflating: WTeams.csv              

==================================================
========== ANÁLISIS DE DATOS MASCULINO ===========
==================================================
1. Temporada Regular Masculino
Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT	WFGM	WFGA	...	LFGA3	LFTM	LFTA	LOR	LDR	LAst	LTO	LStl	LBlk	LPF
0	2003	10	1104	68	1328	62	N	0	27	58	...	10	16	22	10	22	8	18	9	2	20
1	2003	10	1272	70	1393	63	N	0	26	62	...	24	9	20	20	25	7	12	8	6	16
2	2003	11	1266	73	1437	61	N	0	24	58	...	26	14	23	31	22	9	12	2	5	23
3 rows × 34 columns




Filas: 118882 | Columnas: 34
Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

Estadísticas descriptivas:
Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT	WFGM	WFGA	...	LFGA3	LFTM	LFTA	LOR	LDR	LAst	LTO	LStl	LBlk	LPF
count	118882.0	118882.0	118882.0	118882.0	118882.0	118882.0	118882	118882.0	118882.0	118882.0	...	118882.0	118882.0	118882.0	118882.0	118882.0	118882.0	118882.0	118882.0	118882.0	118882.0
unique	NaN	NaN	NaN	NaN	NaN	NaN	3	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
top	NaN	NaN	NaN	NaN	NaN	NaN	H	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3 rows × 34 columns


2. Torneo NCAA Masculino
Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT
0	1985	136	1116	63	1234	54	N	0
1	1985	136	1120	59	1345	58	N	0
2	1985	136	1207	68	1250	43	N	0


Filas: 2518 | Columnas: 8
Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

Estadísticas descriptivas:
Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT
count	2518.0	2518.0	2518.0	2518.0	2518.0	2518.0	2518	2518.0
unique	NaN	NaN	NaN	NaN	NaN	NaN	1	NaN
top	NaN	NaN	NaN	NaN	NaN	NaN	N	NaN

3. Semillas del Torneo Masculino
Season	Seed	TeamID
0	1985	W01	1207
1	1985	W02	1210
2	1985	W03	1228


Filas: 2626 | Columnas: 3
Columnas: ['Season', 'Seed', 'TeamID']

Estadísticas descriptivas:
Season	Seed	TeamID
count	2626.0	2626	2626.0
unique	NaN	94	NaN
top	NaN	W01	NaN

4. Equipos Masculino
TeamID	TeamName	FirstD1Season	LastD1Season
0	1101	Abilene Chr	2014	2025
1	1102	Air Force	1985	2025
2	1103	Akron	1985	2025


Filas: 380 | Columnas: 4
Columnas: ['TeamID', 'TeamName', 'FirstD1Season', 'LastD1Season']

Estadísticas descriptivas:
TeamID	TeamName	FirstD1Season	LastD1Season
count	380.0	380	380.0	380.0
unique	NaN	380	NaN	NaN
top	NaN	West Georgia	NaN	NaN


==================================================
=========== ANÁLISIS DE DATOS FEMENINO ===========
==================================================
1. Temporada Regular Femenino
Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT	WFGM	WFGA	...	LFGA3	LFTM	LFTA	LOR	LDR	LAst	LTO	LStl	LBlk	LPF
0	2010	11	3103	63	3237	49	H	0	23	54	...	13	6	10	11	27	11	23	7	6	19
1	2010	11	3104	73	3399	68	N	0	26	62	...	21	14	27	14	26	7	20	4	2	27
2	2010	11	3110	71	3224	59	A	0	29	62	...	14	19	23	17	23	8	15	6	0	15
3 rows × 34 columns



Filas: 81708 | Columnas: 34
Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

Estadísticas descriptivas:
Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT	WFGM	WFGA	...	LFGA3	LFTM	LFTA	LOR	LDR	LAst	LTO	LStl	LBlk	LPF
count	81708.0	81708.0	81708.0	81708.0	81708.0	81708.0	81708	81708.0	81708.0	81708.0	...	81708.0	81708.0	81708.0	81708.0	81708.0	81708.0	81708.0	81708.0	81708.0	81708.0
unique	NaN	NaN	NaN	NaN	NaN	NaN	3	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
top	NaN	NaN	NaN	NaN	NaN	NaN	H	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3 rows × 34 columns


2. Torneo NCAA Femenino
Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT
0	1998	137	3104	94	3422	46	H	0
1	1998	137	3112	75	3365	63	H	0
2	1998	137	3163	93	3193	52	H	0


Filas: 1650 | Columnas: 8
Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

Estadísticas descriptivas:
Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT
count	1650.0	1650.0	1650.0	1650.0	1650.0	1650.0	1650	1650.0
unique	NaN	NaN	NaN	NaN	NaN	NaN	3	NaN
top	NaN	NaN	NaN	NaN	NaN	NaN	N	NaN

3. Semillas del Torneo Femenino
Season	Seed	TeamID
0	1998	W01	3330
1	1998	W02	3163
2	1998	W03	3112


Filas: 1744 | Columnas: 3
Columnas: ['Season', 'Seed', 'TeamID']

Estadísticas descriptivas:
Season	Seed	TeamID
count	1744.0	1744	1744.0
unique	NaN	80	NaN
top	NaN	W01	NaN

4. Equipos Femenino
TeamID	TeamName
0	3101	Abilene Chr
1	3102	Air Force
2	3103	Akron


Filas: 378 | Columnas: 2
Columnas: ['TeamID', 'TeamName']

Estadísticas descriptivas:
TeamID	TeamName
count	378.0	378
unique	NaN	378
top	NaN	West Georgia

PREDICCIÓN DE GANADORES POR RONDA - NCAA 2025

Cargando datos para el torneo masculino...

Columnas en cada archivo:
Teams: ['TeamID', 'TeamName', 'FirstD1Season', 'LastD1Season']
Seeds: ['Season', 'Seed', 'TeamID']
Regular: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']
Tourney: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

**************************************************
PREDICCIONES TORNEO MASCULINO 2025
**************************************************

RONDA 1:
  Duke vs Houston
  Probabilidad: 0.56 -> GANADOR: Houston
  ----------------------------------------
  Florida vs Auburn
  Probabilidad: 0.50 -> GANADOR: Florida
  ----------------------------------------
  Michigan St vs St John's
  Probabilidad: 0.50 -> GANADOR: St John's
  ----------------------------------------
  Tennessee vs Alabama
  Probabilidad: 0.51 -> GANADOR: Alabama
  ----------------------------------------
  Wisconsin vs Kentucky
  Probabilidad: 0.58 -> GANADOR: Wisconsin
  ----------------------------------------
  Texas Tech vs Iowa St
  Probabilidad: 0.51 -> GANADOR: Texas Tech
  ----------------------------------------
  Texas A&M vs Maryland
  Probabilidad: 0.56 -> GANADOR: Texas A&M
  ----------------------------------------
  Purdue vs Arizona
  Probabilidad: 0.55 -> GANADOR: Purdue
  ----------------------------------------
  Oregon vs Clemson
  Probabilidad: 0.51 -> GANADOR: Clemson
  ----------------------------------------
  Memphis vs Michigan
  Probabilidad: 0.51 -> GANADOR: Michigan
  ----------------------------------------
  Mississippi vs Missouri
  Probabilidad: 0.53 -> GANADOR: Mississippi
  ----------------------------------------
  Illinois vs BYU
  Probabilidad: 0.52 -> GANADOR: Illinois
  ----------------------------------------
  St Mary's CA vs UCLA
  Probabilidad: 0.55 -> GANADOR: St Mary's CA
  ----------------------------------------
  Kansas vs Marquette
  Probabilidad: 0.55 -> GANADOR: Marquette
  ----------------------------------------
  Louisville vs Connecticut
  Probabilidad: 0.51 -> GANADOR: Connecticut
  ----------------------------------------
  Gonzaga vs Mississippi St
  Probabilidad: 0.55 -> GANADOR: Mississippi St
  ----------------------------------------
  Oklahoma vs Creighton
  Probabilidad: 0.56 -> GANADOR: Creighton
  ----------------------------------------
  Baylor vs Georgia
  Probabilidad: 0.51 -> GANADOR: Georgia
  ----------------------------------------
  Utah St vs Vanderbilt
  Probabilidad: 0.53 -> GANADOR: Utah St
  ----------------------------------------
  New Mexico vs Arkansas
  Probabilidad: 0.56 -> GANADOR: New Mexico
  ----------------------------------------
  San Diego St vs Texas
  Probabilidad: 0.56 -> GANADOR: San Diego St
  ----------------------------------------
  VCU vs Xavier
  Probabilidad: 0.54 -> GANADOR: VCU
  ----------------------------------------
  Drake vs North Carolina
  Probabilidad: 0.59 -> GANADOR: Drake
  ----------------------------------------
  Liberty vs McNeese St
  Probabilidad: 0.53 -> GANADOR: Liberty
  ----------------------------------------
  Colorado St vs UC San Diego
  Probabilidad: 0.53 -> GANADOR: Colorado St
  ----------------------------------------
  High Point vs Yale
  Probabilidad: 0.52 -> GANADOR: Yale
  ----------------------------------------
  Akron vs Grand Canyon
  Probabilidad: 0.51 -> GANADOR: Akron
  ----------------------------------------
  Lipscomb vs Montana
  Probabilidad: 0.54 -> GANADOR: Montana
  ----------------------------------------
  UNC Wilmington vs Troy
  Probabilidad: 0.56 -> GANADOR: UNC Wilmington
  ----------------------------------------
  Bryant vs Wofford
  Probabilidad: 0.52 -> GANADOR: Wofford
  ----------------------------------------
  Robert Morris vs NE Omaha
  Probabilidad: 0.58 -> GANADOR: Robert Morris
  ----------------------------------------
  Mt St Mary's vs American Univ
  Probabilidad: 0.53 -> GANADOR: American Univ
  ----------------------------------------
  St Francis PA vs Alabama St
  Probabilidad: 0.51 -> GANADOR: Alabama St
  ----------------------------------------
  SIUE vs Norfolk St
  Probabilidad: 0.51 -> GANADOR: SIUE
  ----------------------------------------

RONDA 2:
  Houston vs Clemson
  Probabilidad: 0.51 -> GANADOR: Clemson
  ----------------------------------------
  Florida vs Michigan
  Probabilidad: 0.51 -> GANADOR: Michigan
  ----------------------------------------
  St John's vs Mississippi
  Probabilidad: 0.51 -> GANADOR: St John's
  ----------------------------------------
  Alabama vs Illinois
  Probabilidad: 0.51 -> GANADOR: Alabama
  ----------------------------------------
  Wisconsin vs St Mary's CA
  Probabilidad: 0.53 -> GANADOR: Wisconsin
  ----------------------------------------
  Texas Tech vs Marquette
  Probabilidad: 0.51 -> GANADOR: Marquette
  ----------------------------------------
  Texas A&M vs Connecticut
  Probabilidad: 0.54 -> GANADOR: Texas A&M
  ----------------------------------------
  Purdue vs Mississippi St
  Probabilidad: 0.51 -> GANADOR: Mississippi St

Desarrollado por: J.E. Carmona Alvarez & J. Ortiz- Aguilar
