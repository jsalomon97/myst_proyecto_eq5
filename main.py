
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
import scipy.stats as st
from datetime import timedelta
import matplotlib.pyplot as plt
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import warnings
warnings.filterwarnings("ignore")
# Librerias para Aspectos Estadisticos
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.tsa.stattools import adfuller
# Librerias para Aspectos Comouracionales
# Librerias para BackTest
from IPython import display
# Importar Data y Funciones
import functins as fn
import data as dt
from data import *
from functions import *

## ASPECTOS FINANCIEROS ##

# Hacemos Compresion de listas para correr la funcion de la descarga de datos para cada uno de los comunicados
# Haremos una lista de DataFrames para almacenar los datos de cada semana.
list_dfs = [Datos_Semana(df.iloc[a,0],1440,1440,"M1") for a in range(len(df))] # 17 DataFrames en la lista

# Ahora hacemos unas graficas para los comunicados
plt.figure(figsize=(20,30))
for i in range(len(df)):
    plt.subplot(8,2,i+1)
    plt.plot(list_dfs[i]['TimeStamp'], list_dfs[i]['Mid'], label='Historico')
    # Asignaremos un color a la línea vertical para diferencial los Consensos de cada uno.
    # Consenso Mayor a Previo
    if df.iloc[i,2] > df.iloc[i,3]:
        c = 'g'
    # Consenso Menor a Previo
    elif df.iloc[i,2] < df.iloc[i,3]:
        c = 'r'
    # Consenso Igual a Previo
    else:
        c = 'k'
    plt.axvline(df.iloc[i,0], color=c, label='Comunicado')
    plt.title('Precio del XAU_USD durante la semana del comunicado del: ' + str(df.iloc[i,0]))
    plt.xlabel('Fecha')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.subplots_adjust(hspace = 0.5)
plt.show()

# Serie de Timpo del Valor Actual
plt.plot(df['DateTime'], df['Actual'], label='Historico Indicador')
plt.title('Valor "Actual" del indicador FED INTEREST RATE DECISION')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()
plt.show()

## ASPECTOS ESTADISTICOS ##

# relacion Lineal entre las observaciones y la distancia de tiempo k de los rezagos
# Lo haremos con un rezago
fn.relacion_lineal()

# https://towardsdatascience.com/heteroscedasticity-is-nothing-to-be-afraid-of-730dd3f7ca1f

df2 = df
df2 = df2.set_index('DateTime')
df2['Time_Period'] = range(1, len(df2)+1)
df2['LOG_Actual'] = np.log(df2['Actual'])
#Create a new mpyplot figure to plot into
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title('Actual')
plt.xlabel('Time Period')
plt.ylabel('Price Index')
plt.plot(df2['Time_Period'], df2['Actual'], 'bo-', label='Actual')
plt.subplot(1,3,2)
plt.title('LOG_Actual')
plt.xlabel('Time Period')
plt.ylabel('Price Index')
plt.plot(df2['Time_Period'], df2['LOG_Actual'], 'bo-', label='LOG_Actual')
expr = 'LOG_Actual ~ Time_Period'
olsr_results = smf.ols(expr, df2).fit()
plt.subplot(1,3,3)
plt.title('Residual errors against Time_Period')
plt.xlabel('Time_Period')
plt.ylabel('Residual Errors')
plt.plot(df2['Time_Period'], olsr_results.resid, 'go-', label='Residual Errors')
plt.show()
# Homo es constante:
# Hetero es no constante:
print('Los errores no son constantes, por tanto es Heterocedástica')
df2['SQ_RESID'] = np.power(olsr_results.resid, 2.0)
df2['SQ_Time_Period'] = np.power(df2['Time_Period'], 2.0)
aux_expr = 'SQ_RESID ~ Time_Period + SQ_Time_Period'
y, X = dmatrices(aux_expr, df2, return_type='dataframe')
X = sm.add_constant(X)
aux_olsr_results = sm.OLS(y, X).fit()
print(aux_olsr_results.summary())
# R-squared: The model has been able to explain only 0.8% of the variance in the squared residuals,
#            indicating a rather poor fit.
# F-statistic: The very high p-value of 0.593 makes us accept the null hypothesis of the F-test that the
#              model’s parameter values are not jointly significant. This model is no better than a mean model.

# DATA
measurements_1 = np.array(df.iloc[:,1])
# Parametros de las distribuciones
dist_1 = 'norm'
# GRAFICOS
plt.figure(figsize=(15,5))
# PRIMERA DISTRIBUCIÓN
plt.subplot(1,2,1)
plt.hist(measurements_1,density=True)  # Histograma
params = getattr(st, dist_1).fit(measurements_1)  # Parametros de la distribución que mejor se acomodan
x = np.arange(measurements_1.min(), measurements_1.max()+0.25, 0.25)  # Espacio en X
y = getattr(st, dist_1).pdf(x, *params)  # Graficar PDF de la distribución que queremos probar
plt.plot(x,y,'r--')  # si siguiera la distribucion normal , los datos se ajustarian a la línea roja.
plt.ylabel('Probability')
plt.grid()
# gráfica de Q-Q entre mis datos y la curva que quiero probar que sigue mi distribución (dist): Usando scipy
plt.subplot(1,2,2)
grap2 = st.probplot(measurements_1, dist=dist_1, sparams=getattr(st, dist_1).fit(x)[:], plot=plt.subplot(1,2,2), fit=True)
plt.grid()
plt.title('Usando paquete scipy')
print('Parátrós de la distribución normal que mejor se acomodan a los datos (mu,sigma).')
print(getattr(st, dist_1).fit(x)[:])
plt.show()
print('Son pocos datos para poder decir que sigue una distribución normal pero se asemeja a una con curtosis de tipo "mesokurtica" (achatada, alargada en las colas).')


print('Basta observar la gráfica de la serie de tiempo para observar que no es estacional.')
# Serie de Timpo del Valor Actual
plt.plot(df['DateTime'], df['Actual'], label='Historico Indicador')
plt.title('Valor "Actual" del indicador FED INTEREST RATE DECISION')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()
plt.show()



X = measurements_1
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
# Entre más negativo sea el valor stadístico (ADF Statistic), Es más probable rechazar la hipotesis nula:
# La data es estacionaria.
print('La serie de tiempo NO es estacionaria.')


# Prueba para datos atípicos:
# El promedio movil puede funcionar como un filtro, aplicandolo como una medida de la relación
# tendencia/ruido
t = df.iloc[:,0]
y_test = pd.Series(df.index + 1)
m = len(t)
T = list(np.ones(m))
for i in np.arange(m):
    T[i] = np.mean(y_test[np.max([0,i-m]):(i+1)])
plt.figure()
plt.plot(t, T, label = 'Tendencia')
N = y_test - T
plt.plot(t,N, label='Ruido')
plt.title('Detección de Datos Atípicos')
plt.legend()
plt.show()
np.std(N)
# https://stats.stackexchange.com/questions/427327/simple-outlier-detection-for-time-series
print('Como no se observan saltos en la línea naranja, no hay datos atípicos del indicador.')


## ASPECTOS COMPUTACIONALES ##
df['Clasificación'] = 0
for i in range(len(df)):
    if df.iloc[i,1] >= df.iloc[i,2] >= df.iloc[i,3]:
        df.iloc[i,5] = 'A'
    elif df.iloc[i,1] >= df.iloc[i,2] < df.iloc[i,3]:
        df.iloc[i,5] = 'B'
    elif df.iloc[i,1] < df.iloc[i,2] >= df.iloc[i,3]:
        df.iloc[i,5] = 'C'
    elif df.iloc[i,1] < df.iloc[i,2] < df.iloc[i,3]:
        df.iloc[i,5] = 'D'
plt.hist(df.Clasificación)
plt.xlabel('Clasificación')
plt.ylabel('Frecuencia')
plt.title('Escenarios de Ocurrencia')
plt.show()

# De los datos descargados de velas de 1 minuto eliminamos todos los valores de 15 minutos antes del comunicado
list_dfs_2 = [list_dfs[a][list_dfs[a]['TimeStamp'] >= str(df['DateTime'][a]-timedelta(minutes=15))] for a in range(len(df))]
# De los datos anteriores eliminamos todos los valores de 15 minutos despúes del comunicado
list_dfs_3 = [list_dfs_2[a][list_dfs_2[a]['TimeStamp'] <= str(df['DateTime'][a]+timedelta(minutes=15))] for a in range(len(df))]
# Ahora hay que reacomodar los índices:
list_dfs_4 = [list_dfs_3[a].reset_index(drop=True) for a in range(len(list_dfs_3))]

direccion = []
pips_alcistas = []
pips_bajistas = []
volatilidad = []
for a in range(len(list_dfs_4)):
    # Se definen los números de los indices porque OANDA no tiene todas las velas de 1 minuto para todos los periodos
    # en otras palabras, faltan velas de 1 minuto.
    t_0_idx = list_dfs_4[a].index[list_dfs_4[a]['TimeStamp'] == str(df['DateTime'][a])][0]  # indice en tiempo 0
    t_30_idx = len(list_dfs_4[a])-1  # indice en tiempo 15
    # el indice en tiempo -15 se obia como "0"
    direccion.append((list_dfs_4[a]['Close'][t_30_idx] - list_dfs_4[a]['Open'][t_0_idx])*10000)
    pips_alcistas.append((max(list_dfs_4[a]['High'][t_0_idx:t_30_idx]) - list_dfs_4[a]['Open'][t_0_idx])*10000)
    pips_bajistas.append((list_dfs_4[a]['Open'][t_0_idx] - min(list_dfs_4[a]['Low'][t_0_idx:t_30_idx]))*10000)
    volatilidad.append((max(list_dfs_4[a]['High'][0:t_30_idx]) - min(list_dfs_4[a]['Low'][0:t_30_idx]))*10000)
df['direccion'] = [int(round(num, 0)) for num in direccion]
df['pips_alcistas'] = [int(round(num, 0)) for num in pips_alcistas]
df['pips_bajistas'] = [int(round(num, 0)) for num in pips_bajistas]
df['volatilidad'] = [int(round(num, 0)) for num in volatilidad]
print(df)

## BACKTEST ##
# Crear df escenarios
Cap_Ini = 100000
Riesgo_Max = 1000
df_escenarios = df[['DateTime', 'Clasificación', 'direccion','pips_alcistas', 'pips_bajistas', 'volatilidad']]
df_escenarios['direccion'] = df['direccion'].apply(lambda x: -1 if x < 0 else 1)
# Ahora acomodamos los DF para que vayan desde la fecha más antigua, a la más reciente
df_escenarios = df_escenarios.sort_values(by='DateTime')
df_escenarios_train = df_escenarios.iloc[0:8,:].reset_index(drop=True)
df_escenarios_test = df_escenarios.iloc[8:,:].reset_index(drop=True)
print('El análisis de estos datos de entrenamiento está cesgado porque solo hay un tipo de clasificación (A)')
print(df_escenarios_train)

# Crear df decisiones
# Si aumenta la tasa la plata se deprecia, si disminuye la tasa, la plata se aprecia.
df_escenarios = pd.DataFrame()
df_escenarios['escenarios'] = ['A', 'B', 'C', 'D']
df_escenarios['operacion'] = ['compra', 'venta', 'compra', 'venta']
df_escenarios['sl'] = [100, 100, 100, 100]
df_escenarios['tp'] =[200, 200, 200, 200]
df_escenarios['volumen'] = [4500, 4500, 4500, 4500]
print(df_escenarios)

## Algotimos Geneticos: Con restricciones PSO
## Condiciones Iniciales
num_p = 10 # numero de pobladores
n_ite = 100 # numero de iteraciones
# TP
TP_max = 300
TP_min = 100
# SL
SL_max = 150
SL_min = 50
# V
V_max = 1500
V_min = 800
## Take Profit
x1p = np.random.randint(TP_min,TP_max,num_p) # posicion inicial (de a cant)
x1pg = 0  # pocision inicial del global
x1pl = x1p # valores iniciales de los mejores locales
vx1 = np.zeros(num_p) # velocidad inicial de las particulas
## Stop Loss
x2p = np.random.randint(SL_min,SL_max,num_p) # posicion inicial (de a cant)
x2pg = 0  # pocision inicial del global
x2pl = x2p # valores iniciales de los mejores locales
vx2 = np.zeros(num_p) # velocidad inicial de las particulas
## Volumen
x3p = np.random.randint(V_min,V_max,num_p) # posicion inicial (de a cant)
x3pg = 0  # pocision inicial del global
x3pl = x3p # valores iniciales de los mejores locales
vx3 = np.zeros(num_p) # velocidad inicial de las particulas
## Parametros Iniciales
fxpg = 0 # Desempeño inicial del mejor global
fxpl = np.ones(num_p)*fxpg # desempeño de los mejores locales
c1 = 0.75 # velocidad de convergencia al mejor global
c2 = 0.75 # velocidad de convergencia al mejor local
# como el PSO busca maximizar, la penalizacion disminuye
a = 1000
x_iter = list()  #  Aqui guardamos los datos de x para la grafica de desempeño
y_iter = list()  #  Aqui guardamos los datos de y para la grafica de desempeño
## Funcionamiento
for k in range(n_ite): # iteraciones del algoritmo
    fx = np.ones(num_p)
    for j in range(num_p): # iteraciones del algoritmo
        fx[j] = fun_opt_ganacia_fundamental(df_escenarios_train,x3p[j],x1p[j],x2p[j]) - a*max(x1p[j]-TP_max,0) + a*min(x1p[j]-TP_min,0) - a*max(x2p[j]-SL_max,0) + a*min(x2p[j]-SL_min,0) - a*max(x3p[j]-V_max,0) + a*min(x3p[j]-V_min,0)
    # Determinar el máximo global
    val = max(fx)
    ind = list(fx).index(max(fx))
    if val>fxpg:
        x1pg = x1p[ind] # guardar la posicion del mejor
        x2pg = x2p[ind] # guardar la posicion del mejor
        x3pg = x3p[ind] # guardar la posicion del mejor
        fxpg = val # guardar el valor del mejor
    # Determinar el máximo local
    for p in range(num_p):
        if fx[p]>fxpl[p]:
            fxpl[p] = fx[p] # remplazo el valor del mejor local
            x1pl[p] = x1p[p] # remplazo la posicion del mejor local
            x2pl[p] = x2p[p] # remplazo la posicion del mejor local
            x3pl[p] = x3p[p] # remplazo la posicion del mejor local
    fx = fun_opt_ganacia_fundamental(df_escenarios_train,x3pg,x1pg,x2pg)
    # Ecuaciones de Movimineto
    vx1 = vx1+np.dot(c1*np.random.rand(),x1pg-x1p)+np.dot(c2*np.random.rand(),x1pl-x1p) # la velocidad
    x1p = np.round(vx1 + x1p,0).astype(int)  # Como son pips tienen que ser números enteros
    vx2 = vx2+np.dot(c1*np.random.rand(),x2pg-x2p)+np.dot(c2*np.random.rand(),x2pl-x2p) # la velocidad
    x2p = np.round(vx2 + x2p,0).astype(int)  # Como son pips tienen que ser números enteros
    vx3 = vx3+np.dot(c1*np.random.rand(),x3pg-x3p)+np.dot(c2*np.random.rand(),x3pl-x3p) # la velocidad
    x3p = np.round(vx3 + x3p,0).astype(int)  # Como son volumen tienen que ser números enteros
    x_iter.append(k+1)
    y_iter.append(fx)
print('Mejor Resultado después de ' + str(n_ite) + ' iteraciones:')
print('Max profit:' + str(fun_opt_ganacia_fundamental(df_escenarios_train,x3pg,x1pg,x2pg)))
print('Take Profit óptimo:' + str(x1pg))
print('Stop Loss óptimo:' + str(x2pg))
print('Volumen óptimo:' + str(x3pg))
plt.plot(x_iter,y_iter)
plt.title('Gráfica de Convergencia')
plt.xlabel('iteraciones')
plt.ylabel('Max profit')
plt.show()


print('Resultados de la Optimización con los Datos de Entrenamiento')
exp_ = fun_opt_ganacia_fundamental_disp(df_escenarios_train, x3pg, x1pg, x2pg) # esperimento de la función
print('la ganancia es de: ' + str(exp_[1]) + ' USD.')
plt.figure(figsize=(15, 5))
plt.plot(exp_[0]['DateTime'], exp_[0]['Capital_Acm'])
plt.title('evolucion de capital')
plt.xlabel('Tiempo')
plt.ylabel('Capital Acumulado')
plt.yticks(range(100000, 130000+1,5000))
plt.show()

print('Resultados de la Optimización con los Datos de Prueba')
exp_ = fun_opt_ganacia_fundamental_disp(df_escenarios_test,x3pg,x1pg,x2pg) # esperimento de la función
print('la ganancia es de: ' + str(exp_[1]) + ' USD.')
plt.figure(figsize=(15, 5))
plt.plot(exp_[0]['DateTime'], exp_[0]['Capital_Acm'])
plt.title('evolucion de capital')
plt.xlabel('Tiempo')
plt.ylabel('Capital Acumulado')
plt.yticks(range(100000, 130000+1,5000))
plt.show()

## MAD ##
print('Medidas de Atribución al Desempeño')
df_data_MAD = exp_[0][['DateTime', 'Capital', 'Capital_Acm']]
# Agregamos al DF la columna con capital incial de 100,000 para calcular el DrawDown y DrawUp
data_start = []
data_start.insert(0, {'DateTime':df_data_MAD.iloc[0, 0]-timedelta(minutes=1), 'Capital': 0, 'Capital_Acm': 100000})
df_data_MAD = pd.concat([pd.DataFrame(data_start), df_data_MAD], ignore_index=True)
MAD = f_estadisticas_mad(df_data_MAD)
print(MAD)






