
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Analisis dundamental del XAG_USD y la tasa de interés                                      -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: eremarin45, luismaria8992ramirez, jsalomon97                                                -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/luismaria8992ramirez/myst_proyecto_eq5                                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# Librerias para Aspectos Financieros
import numpy as np
import pandas as pd
from datetime import timedelta
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import warnings
warnings.filterwarnings("ignore")
# Librerias para Aspectos Estadisticos
# Librerias para Aspectos Comouracionales
# Librerias para BackTest
from IPython import display

Oanda_Tk = '239520fe9ffa6481d40db605a43d798e-7ff3d9a09b0b4e63039e35b491bb169b'  # Token de OANDA
api = API(access_token=Oanda_Tk)  # Inicializar API de OANDA

# Crear Una Funcion donde yo le doy una fecha del comunicado (miercoles) y la funcion me da el histórico de precios
# de esa fecha más/menos 1 día. Asi podremos analizar mejor los datos.
def Datos_Semana(Fecha_Comunicado,mins1,mins2,granu):
    # Definimos el rango de fechas en el que descargaremos los datos:
    fecha_media = Fecha_Comunicado  # Fecha de el comunicado
    fecha_inicio = fecha_media - timedelta(minutes=mins1)  # 24 horas * 60 minutos
    fecha_fin = fecha_media + timedelta(minutes=mins2)
    Fecha_1 = fecha_inicio.strftime('%Y-%m-%dT%H:%M:%S')  # Fecha 1
    Fecha_2 = fecha_fin.strftime('%Y-%m-%dT%H:%M:%S')  # Fecha 2
    list_df = []  # En esta lista metermos las velas del diccionario para hacer un DataFrame
    Request = instruments.InstrumentsCandles(instrument="XAG_USD" ,   # Instrumento
                                             params={"granularity": granu, # Tamaño de Velas 5 mins
                                                     # "price": "M",
                                                     "dailyAlignment": 9,  # A partir de que hora se alinean las velas
                                                     "alignmentTimezone": "America/New_York",  # Zona horaria
                                                     "from": Fecha_1, "to": Fecha_2})  # Fechas de Martes-Jueves
    Dict_Semana = api.request(Request)
    for i in range(len(Dict_Semana['candles'])-1):
        list_df.append({'TimeStamp': Dict_Semana['candles'][i]['time'],
                        'Open': Dict_Semana['candles'][i]['mid']['o'],
                        'High': Dict_Semana['candles'][i]['mid']['h'],
                        'Low': Dict_Semana['candles'][i]['mid']['l'],
                        'Close': Dict_Semana['candles'][i]['mid']['c']})
    # Creamos y Acomodamos el DataFrame
    Df_historico_Semana = pd.DataFrame(list_df)  # Convertimos la lista a DataFrame
    Df_historico_Semana = Df_historico_Semana[['TimeStamp', 'Open', 'High', 'Low', 'Close']]  # DF nombre de columnas
    Df_historico_Semana['TimeStamp'] = pd.to_datetime(Df_historico_Semana['TimeStamp'])  # A timestamp
    Df_historico_Semana['Open'] = pd.to_numeric(Df_historico_Semana['Open'])  # a numerico
    Df_historico_Semana['High'] = pd.to_numeric(Df_historico_Semana['High'])  # a numerico
    Df_historico_Semana['Low'] = pd.to_numeric(Df_historico_Semana['Low'])  # a numerico
    Df_historico_Semana['Close'] = pd.to_numeric(Df_historico_Semana['Close'])  # a numerico
    Df_historico_Semana['Mid'] = (Df_historico_Semana['High'] + Df_historico_Semana['Low'])/2
    return Df_historico_Semana


# Crear df Backtest
# Esta función Usa los parametros de Take Profit, Stop Loss y Volumen y va iterando para calcular la ganancia
def fun_opt_ganacia_fundamental(df_backtest,vol,tp,st):
    df_backtest['Operación'] = df_backtest['Clasificación'].apply(lambda x: 'venta' if (x == 'A' or x == 'C') else 'compra')
    df_backtest['Volumen'] = vol*100 # Consideramos apalancamiento 100 a 1 para poder hacer esto.
    df_backtest['Resultado'] = 0
    df_backtest['Pips'] = 0
    df_backtest['Capital'] = 0
    df_backtest['Capital_Acm'] = 100000
    for i in range(len(df_backtest)):
        if (df_backtest['Clasificación'][i] == 'A' or df_backtest['Clasificación'][i] == 'C'):
            if (df_backtest['pips_alcistas'][i] >= df_backtest['pips_bajistas'][i]):
                df_backtest['Resultado'][i] = 'ganada'
                df_backtest['Pips'][i] = tp if df_backtest['pips_alcistas'][i] > tp else df_backtest['pips_alcistas'][i]
            else:
                df_backtest['Resultado'][i] = 'perdida'
                df_backtest['Pips'][i] = st if df_backtest['pips_bajistas'][i] > st else df_backtest['pips_bajistas'][i]
        else:
            if (df_backtest['pips_alcistas'][i] <= df_backtest['pips_bajistas'][i]):
                df_backtest['Resultado'][i] = 'ganada'
                df_backtest['Pips'][i] = tp if df_backtest['pips_bajistas'][i] > tp else df_backtest['pips_bajistas'][i]
            else:
                df_backtest['Resultado'][i] = 'perdida'
                df_backtest['Pips'][i] = st if df_backtest['pips_alcistas'][i] > st else df_backtest['pips_alcistas'][i]
        df_backtest['Capital'][i] = (df_backtest['Pips'][i]/10000)*df_backtest['Volumen'][i] if df_backtest['Resultado'][i] == 'ganada' else -(df_backtest['Pips'][i]/10000)*df_backtest['Volumen'][i]
        df_backtest['Capital_Acm'][i] = 100000 + sum(df_backtest.iloc[0:i+1,10])
    df_result = df_backtest[['DateTime', 'Clasificación', 'Operación', 'Volumen', 'Resultado', 'Pips', 'Capital', 'Capital_Acm']]
    # display.display(df_result)
    return df_result.iloc[-1,-1]-100000


# Misma Función que la anterior pero Imprime el DataFrame para que en el anterior no imprima en cada itearacion.
def fun_opt_ganacia_fundamental_disp(df_backtest,vol,tp,st):
    df_backtest['Operación'] = df_backtest['Clasificación'].apply(lambda x: 'venta' if (x == 'A' or x == 'C') else 'compra')
    df_backtest['Volumen'] = vol*100 # Consideramos apalancamiento 100 a 1 para poder hacer esto.
    df_backtest['Resultado'] = 0
    df_backtest['Pips'] = 0
    df_backtest['Capital'] = 0
    df_backtest['Capital_Acm'] = 100000
    for i in range(len(df_backtest)):
        if (df_backtest['Clasificación'][i] == 'A' or df_backtest['Clasificación'][i] == 'C'):
            if (df_backtest['pips_alcistas'][i] >= df_backtest['pips_bajistas'][i]):
                df_backtest['Resultado'][i] = 'ganada'
                df_backtest['Pips'][i] = tp if df_backtest['pips_alcistas'][i] > tp else df_backtest['pips_alcistas'][i]
            else:
                df_backtest['Resultado'][i] = 'perdida'
                df_backtest['Pips'][i] = st if df_backtest['pips_bajistas'][i] > st else df_backtest['pips_bajistas'][i]
        else:
            if (df_backtest['pips_alcistas'][i] <= df_backtest['pips_bajistas'][i]):
                df_backtest['Resultado'][i] = 'ganada'
                df_backtest['Pips'][i] = tp if df_backtest['pips_bajistas'][i] > tp else df_backtest['pips_bajistas'][i]
            else:
                df_backtest['Resultado'][i] = 'perdida'
                df_backtest['Pips'][i] = st if df_backtest['pips_alcistas'][i] > st else df_backtest['pips_alcistas'][i]
        df_backtest['Capital'][i] = (df_backtest['Pips'][i]/10000)*df_backtest['Volumen'][i] if df_backtest['Resultado'][i] == 'ganada' else -(df_backtest['Pips'][i]/10000)*df_backtest['Volumen'][i]
        df_backtest['Capital_Acm'][i] = 100000 + sum(df_backtest.iloc[0:i+1,10])
    df_result = df_backtest[['DateTime', 'Clasificación', 'Operación', 'Volumen', 'Resultado', 'Pips', 'Capital', 'Capital_Acm']]
    display.display(df_result)
    return [df_result, df_result.iloc[-1,-1]-100000]


# FUN f_estadisticas_mad
# Data con cada metrica para valores diarios
# Sharpe,DrawDown & DrawUp
def f_estadisticas_mad(param_data):
    df = param_data
    # rp: Promedio de los rendimientos logarítmicos de profit_acm_d
    ren_log = np.log(df['Capital_Acm'] / df['Capital_Acm'].shift(1)).iloc[1:].cumsum()
    rp = np.mean(ren_log)  # promedio de log calculados
    sdp = ren_log.std()  # desv de los rendimientos
    rf = .05 / 8  # rf: 5% y estamos usando 8 periodos anuales
    # Sharpe Ratio
    sharpe = (rp - rf) / sdp
    # DrawDown
    # Minusvalia máxima que se registró en la evolución de los valores (de 'profit_acm_d')
    # Fecha Inicial", "Fecha Final", "DrawDown $ (capital)
    DD = 0
    for i in range(len(df)):
            for j in range(i,len(df)):
                val = min([df.iloc[j,2]-df.iloc[i,2],0])
                DD = min(DD,val)
    # DrawUp
    # Plusvalía máxima que se registró en la evolución de los valores (de 'profit_acm_d')
    # Fecha Inicial", "Fecha Final", "DrawUp $ (capital)
    DU = 0
    for i in range(len(df)):
            for j in range(i,len(df)):
                val = max([df.iloc[j,2]-df.iloc[i,2],0])
                DU = max(DU,val)
    mad = {'Metrica': ['Sharpe', 'Drawdown_capi', 'Drawup_capi'],
           'Valor': [sharpe, DD, DU],
           'Descripción': ['Sharpe Ratio', 'DrawDown de Capital', 'DrawUp de Capital']}
    df_mad = pd.DataFrame(mad)
    return df_mad
