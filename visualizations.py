
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Analisis dundamental del XAG_USD y la tasa de interés                                      -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: eremarin45, luismaria8992ramirez, jsalomon97                                                -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/luismaria8992ramirez/myst_proyecto_eq5                                                               -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

def serie_tiempo_va(df):
    plt.plot(df['DateTime'], df['Actual'], label='Historico Indicador')
    plt.title('Valor "Actual" del indicador FED INTEREST RATE DECISION')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()

def fig_heros(df2):
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

def datos_atipicos(df):
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
