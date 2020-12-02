
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Analisis dundamental del XAG_USD y la tasa de interés                                      -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: eremarin45, luismaria8992ramirez, jsalomon97                                                -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/luismaria8992ramirez/myst_proyecto_eq5                                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import pandas as pd

ind_dir = 'Files/' + 'Fed Interest Rate Decision - United States.txt'  # directorio del indicador
df = pd.read_csv( ind_dir, sep=",", header=0)  # Importacion
df['DateTime']= pd.to_datetime(df['DateTime'])  # Fechas en formato fecha
df = df[df['DateTime'] > '28/02/2018']  # Filtrar ultimos 2 años
# Columnas: 'DateTime', 'Actual', 'Consensus', 'Previous', 'Revised'
df.head(3)
