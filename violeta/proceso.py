
# .. ................................................................................... .. #
# .. Proyecto: UrbanLab - Plataforma de ayuda para micro y pequeñas empresas             .. #
# .. Archivo: proceso.py - funciones de procesamiento general de datos                   .. #
# .. Desarrolla: ITERA LABS, SAPI de CV                                                  .. #
# .. Licencia: Todos los derechos reservados                                             .. #
# .. Repositorio: https://github.com/IFFranciscoME/Urban_Lab.git                         .. #
# .. ................................................................................... .. #


# Importing and initializing main Python libraries
import math
import pandas as pd
import numpy as np
import datos as dat
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from scipy.stats import shapiro

from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#%%
# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Calculate metric
# -- ------------------------------------------------------------------------------------ -- #
def metric_quantification(df_data, conditions, metric_column):
	"""
    Parameters
    ---------
    :param:
        df_data: DataFrame : datos en DF

    Returns
    ---------
    :return:
        df: DataFrame : Datos del archivo

    Debuggin
    ---------
        df_data = read_file(ent.path, ent.sheet)
	"""
	# Columns names
	list_columns = list(conditions.keys())
	# Conditions (dicts)
	list_dict_conditions = list(conditions.values())
	# List of lists with answers
	answer = [[f_condition(
							list_dict_conditions[k], 
							   df_data[list_columns[k]][i]
							   ) 
							for i in range(len(df_data))
							] 
					for k in range(len(list_columns))
					]
	# sum all
	metric = pd.DataFrame(answer).sum()
	df = df_data.copy()
	df[metric_column] = metric
	return df


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: According with condition in dict
# -- ------------------------------------------------------------------------------------ -- #
def f_condition(dict_condition, data):
	"""
    Parameters
    ---------
    :param:
        dict_condition: dict : diccionario con condiciones
		data: int or str: dato a comparar
    Returns
    ---------
    :return:
        int: valor de acuerdo a la condicion

    Debuggin
    ---------
        dict_condition = list(ent.conditions_stress.values())[0]
		data = df_data['ventas_porcentaje'][0]
	"""
	# valores que se necesitan poner
	posible_results = list(dict_condition.keys())
	# lista de condiciones
	list_conditions = list(dict_condition.values())
	# Utilizando la funcion para cada condicion
	answer = [type_verification(list_conditions[i], posible_results[i], 
							data) for i in range(len(list_conditions ))]
	
	if answer == [0]*len(answer):
		return 0
	else:
		lista = list(filter(None.__ne__, answer))
		if len(lista) == 0:
			return['error']
		else:
			return lista[0]
		

# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Check what kind of condition is needed
# -- ------------------------------------------------------------------------------------ -- #	
def type_verification(condition, result, data):
	"""
    Parameters
    ---------
    :param:
        condition: tuple or list : contiene las condiciones
		result: int : numero si se cumple la condicion es el resultado
		data: int or str: dato que se esta comparando para cuantificar
		
    Returns
    ---------
    :return:
        answer: int : numero de la metrica

    Debuggin
    ---------
        condition = (0, 25)
		result = 1
		data = 10
		
	"""
	# Si es lista tiene que estar en tal
	if type(condition) == list:
		if data in condition:
			return result
	
	# Si es numerico
	if type(data) != str:
		if math.isnan(data):
			return 0
		# Si es tuple, tiene que estar entre ambos numeros
		if type(condition) == tuple:
			if condition[0] < data and data <= condition[1]:
				return result


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: According with condition in dict
# -- ------------------------------------------------------------------------------------ -- #
def f_predict(p_serie_tiempo):
	
	# meses en el futuro, predecir
	meses = 6
	
	# ------------------------------------------ #
	# Primero intentar con una regresión lineal
	# ------------------------------------------ #
	
	# Separar la informacion que se tiene de la serie de tiempo en y
	y_o = np.array(p_serie_tiempo)
	x_o = np.arange(len(p_serie_tiempo))
	
	# Acomodarla de la forma que el modelo necesita
	x = x_o.reshape((len(x_o),1))
	y = y_o.reshape((len(y_o),1))
	
	# Crear el modelo
	modelo = linear_model.LinearRegression()
	
	# Pasar nuestros datos por el modelo
	modelo.fit(x, y)
	
	# De acuerdo al modelo, calcular y
	y_pred = modelo.predict(x)
	
	# R2 de sus residuales
	r_2 = r2_score(y, y_pred)
	
	if r_2 > 0.8:
		# sumar a la x ultima
		value = x_o[-1]+meses
		# predecir
		prediction = modelo.predict(value.reshape((1,1)))
		
		return [p_serie_tiempo[len(p_serie_tiempo)-1], prediction[0][0]]
	
	else: 
	# ------------------------------------------ #
	# Segundo intentar modelar con SARIMA
	# ------------------------------------------ #
	
	# Empezar checando si es estacionaria
		def check_stationarity(data):
		    # Usar dicky fuller
			test_results = sm.tsa.stattools.adfuller(data)
		    # Cuando se cumple esto es estacionaria la serie orginal
			if test_results[0] < 0 and test_results[1] <= 0.05:
				lags = 0
	        
			# Cuando no se cumple se debe diferenciar para que sea estacionaria
			else:
				for i in range(3):
		            # Diferenciar datos
					new_data = np.diff(data)
		            
		            # Volver a calcular test dicky fuller
					new_results = sm.tsa.stattools.adfuller(new_data)
		            
		            # Volver a comparar para decidir si es o no estacionaria
					if new_results[0] < 0 and new_results[1] <= 0.05:
						# rezagos necesarios para volverlo estacionario
						lags = i
						break
		            
					else:
						data = new_data
						# solo permitimos 3 rezagos, si aún no lo es, no se modela
						lags = np.nan
			return lags
		# Checar estacionariedad
		d = check_stationarity(p_serie_tiempo)
		
		if np.isnan(d):
			return [p_serie_tiempo[len(p_serie_tiempo)-1]]
		
		else:
			
			# lambda para tomar los coef significativos
			all_significant_coef = lambda x: x if abs(x)>0.5 else None
			
			def significat_lag(all_coef):
				# Tomar los indices de los rezagos
				ind_c = all_coef.index.values
				# Solo los rezagos menores a 7
				sig_i = ind_c[ind_c<7]
				# Nuevos coeficientes
				new_coef = all_coef[all_coef.index.isin(list(sig_i))]
				if len(new_coef) > 1:
					# Tomar los valores absolutos
					abs_coef = new_coef[1:].abs()
					# Buscar el maximo
					max_coef = abs_coef.max()
					# El indice es el rezago al que pertenece
					answer = abs_coef[abs_coef == max_coef[0]].dropna().index[0]
					return answer
				else:
					return 1
			
		    # Calcular coeficientes de fac parcial
			facp = sm.tsa.stattools.pacf(p_serie_tiempo)
			
		    # Pasar lambda y quitar los que no son significativos
			p_s = pd.DataFrame(all_significant_coef(facp[i]) for i in range(len(facp))).dropna()
			
		    # Tomar el primero que sea signiticativo, sera la p de nuestro modelo
			p = significat_lag(p_s)
			
			# --- #
			
		    # Calcular coeficientes de fac 
			fac = sm.tsa.stattools.acf(p_serie_tiempo, fft=False)
			
		    # Pasar lambda y quitar los que no son significativos
			q_s = pd.DataFrame(all_significant_coef(fac[i]) for i in range(len(fac))).dropna()
			
		    # Tomar el primero que sea signiticativo, sera la p de nuestro modelo
			q = significat_lag(q_s)
			
			# Modelo
			arima = sm.tsa.statespace.SARIMAX(p_serie_tiempo,
									 order=(p,d,q),
									 trend = 'c',
									 enforce_stationarity=True, 
									 enforce_invertibility=True)
			arima_fitted = arima.fit()
			
			def check_resid(model_fit):
				# estadístico Ljung – Box.
				colineal = acorr_ljungbox(model_fit.resid, lags=[10])
				# se necesita aceptar H0, es decir p_value debe ser mayor a .05
				colin = True if colineal[1]>0.05 else False
				
				# shapiro test
				normalidad = shapiro(model_fit.resid)
				# si el p-value es menor a alpha, rechazamos la hipotesis de normalidad
				norm = True if normalidad[1]>0.05 else False
			
				# arch test
				heterosced = het_arch(model_fit.resid)
				# p-value menor a 0.05 y concluir que no hay efecto de heteroscedasticidad
				heter = True if heterosced[1]>0.05 else False
				
				return colin, norm, heter
			
			# predecir siguientes 6 meses
			future_prices = arima_fitted.forecast(meses, alpha=0.05)

			return [p_serie_tiempo[len(p_serie_tiempo)-1],
						  future_prices[len(p_serie_tiempo) + meses - 1]]


	
#%%

	


