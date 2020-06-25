
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
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import  adfuller
from statsmodels.tsa.stattools import pacf,acf
from statsmodels.stats.diagnostic import acorr_ljungbox

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

#%%
'''
_ __ __________________________________________________________________________________ __ _ #

  Analisis de series de tiempo
_ __ ____________________________________________________________________________________ __ #
  
'''


from statsmodels.stats.diagnostic import het_arch
from scipy.stats import shapiro
import matplotlib.pyplot as plt

def check_stationarity(data):
    data = data["Actual"]
    test_results = adfuller(data)
    if test_results[0] < 0 and test_results[1] <= 0.05:
        result = True
    else:
        result = False
    results = {'Resultado': result,
               'Test stadisctic': test_results[0],
               'P-value': test_results[1]
               }
    out=results
    if test_results[1]>0.01:
        data_d1 = data.diff().dropna()
        results_d1 = adfuller(data_d1)
        if results_d1[0] < 0 and results_d1[1] <= 0.01:

            results_d1 = {'Resultado':True,
                          'Test stadistic':results_d1[0],
                          'P-value':results_d1[1]

            }
        out={'Datos originales': results,
             'Primera diferencia': results_d1}
    return out


def fit_arima(data):
    test_result = check_stationarity(data)
    if test_result["Resultado"]==True:
        significant_coef = lambda x: x if x>0.5 else None
        #d=0
        p = pacf(data["Actual"])
        p = pd.DataFrame(significant_coef(p[i]) for i in range(0,11))
        idx=0
        for i in range(len(p)):
            if p.iloc[i] !=np.nan:
                idx=i
        p=p.iloc[idx].index

        q = acf(data["Actual"])
        q = pd.DataFrame(significant_coef(q[i]) for i in range(0, 11))
        idx=0
        for i in range(len(q)):
            if q.iloc[i] != np.nan:
                idx = i
        q = q.iloc[idx].index


        #model = ARIMA(data,order=(p,d,q))
        #model.fit(disp=0)
    return [p,q]


def norm_test(data):

    n_test=shapiro(data["Actual"])
    test_results = { 'Test statistic':n_test[0],
                    'P-value':n_test[1] #si el p-value de la prueba es menor a alpha, rechazamos H0
    }
    return test_results
    """
    paramaetros = 2
    J = 90
    grados_libertad = j-p-1
    [freq,x,p]=plt.hist(data,J,density=True)
    pi = st.norm.pdf(x, loc=np.mean(data), scale=np.std(data))
    Ei=x*pi

    x2 = st.chisquare(freq,Ei)
    Chi_est = st.chi2.ppf(q=0.95, df=grados_libertad)
    """
	

def diff_series(data,lags):
    data = data["Actual"].diff(lags).dropna()
    return data


def arch_test(data):
    data=data["Actual"]
    test= het_arch(data)
    results = {"Estadístico de prueba": test[0],
               "P-value":test[1]
    }
    results=pd.DataFrame(results,index=["Resultados"])
    return results


def get_outliers(data):
	#vs.g_AtipicalData(data)
	box = plt.boxplot(data["Actual"])
	bounds = [item.get_ydata() for item in box["whiskers"]]
	datos_atipicos = data.loc[(data["Actual"] > bounds[1][1]) | (data["Actual"] < bounds[0][1])]
	return datos_atipicos
			
#%%


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: According with condition in dict
# -- ------------------------------------------------------------------------------------ -- #
def fit_arima_(data):
	
	def check_stationarity_t(data):
		  
	    # Usar dicky fuller
		test_results = adfuller(data)
	    
	    # Cuando se cumple esto es estacionaria la serie orginal
		if test_results[0] < 0 and test_results[1] <= 0.05:
			lags = 0
			new_data = data
        
		# Cuando no se cumple se debe diferenciar para que sea estacionaria
		else:
			
			for i in range(3):
	            
	            # Diferenciar datos
				new_data = np.diff(data)
				#new_data = new_data_na[~np.isnan(new_data_na)]
	            
	            # Volver a calcular test dicky fuller
				new_results = adfuller(new_data)
	            
	            # Volver a comparar para decidir si es o no estacionaria
				if new_results[0] < 0 and new_results[1] <= 0.05:
	                #print('es estacionaria')
					lags = i
					break
	            
				else:
					data = new_data
					#print('no es estacionaria')
					lags = 3
	                
	    # Regresa los datos diferenciados que son estacionarios
		return lags+1
	
	d = check_stationarity_t(data)
	
    # lambda para tomar los coef significativos
	significant_coef = lambda x: x if x>0.5 else None
	
    # Calcular coeficientes de fac parcial
	facp = pacf(data)
	
    # Pasar lambda y quitar los que no son significativos
	p_s = pd.DataFrame(significant_coef(facp[i]) for i in range(len(facp))).dropna()
	
    # Tomar el primero que sea signiticativo, sera la p de nuestro modelo
	p = p_s.index[0] + 1

    # Calcular coeficientes de fac 
	fac = acf(data, fft=False)
	
    # Pasar lambda y quitar los que no son significativos
	q_s = pd.DataFrame(significant_coef(fac[i]) for i in range(len(fac))).dropna()
	
    # Tomar el primero que sea signiticativo, sera la p de nuestro modelo
	q = q_s.index[0] + 1
	
	# Primer modelo
	print(p,d,q)
	try:
		model = ARIMA(data, order=(p,d,q))
		model_fit = model.fit()
		
		# P values de coeficientes de modelo
		pval = model_fit.pvalues
	
		# Akaike Information Criterion
		m1_aic = model_fit.aic
		# MA term (q) from partial fac
		def pval_signif(pval, p, q):		
			p_s, q_s = [], []
			for i in range(1, len(pval)):
				if i<= p:
					p_s.append(pval[i])
				else:
					q_s.append(pval[i])
			# Ver si son significantes
			if q_s[-1]>0.5:
				return p, q-1
			
			elif p_s[-1]>0.5:
				return p-1, q
			
			else:
				return p, q
		
		p, q = pval_signif(pval, p, q)
		# Nuevo modelo
		new_model = ARIMA(data, order=(p,d,q))
		new_model_fit = new_model.fit()
		# Akaike Information Criterion nuevo
		m2_aic = new_model_fit.aic
		
		return new_model_fit
		
	except:
		#m1_aic, m2_aic = 0, 0
		return np.nan
				

	#return pval
#arima_1 = fit_arima_(time_series[0])


def check_resid(model_fit):
	# estadístico Ljung – Box.
	colineal = acorr_ljungbox(model_fit.resid, lags=[10])
	# se necesita aceptar H0, es decir p_value debe ser mayor a .05
	colin_pv = colineal[1]
	if colin_pv>0.05: 
		col = True
	else:
		col = False
	
	# shapiro test
	normalidad = shapiro(model_fit.resid)
	# si el p-value es menor a alpha, rechazamos la hipotesis de normalidad
	norm_pv = normalidad[1]
	if norm_pv>0.05: 
		norm = True
	else:
		norm = False

	# arch test
	heterosced = het_arch(model_fit.resid)
	# p-value menor a 0.05 y concluir que no hay efecto de heteroscedasticidad
	hete_pv = heterosced[1]
	if hete_pv<0.05: 
		hete = True
	else:
		hete = False
	
	return col, norm, hete
#%%
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
'''
data = time_series[2]
#resDiff = sm.tsa.arma_order_select_ic(data, max_ar=4, max_ma=4, ic='aic', trend='c')
model_2 = ARIMA(data, order=(1,1,1))
model_fit_2 = model_2.fit()

model_fit_2.plot_predict(dynamic=False)
plt.show()
	
'''



	
	
	
	
	
	
	
	
	
	
#%%
#import statsmodels.api as sm

#x = [1 if i%2 == 0 else 6 for i in range(50)]
#eta = np.random.normal(0, 0.01, 50)
#x = x + eta
#res = sm.tsa.stattools.arma_order_select_ic(x, ic=['aic']) 
#print(res.aic_min_order)
#model = sm.tsa.ARMA(x, res.aic_min_order).fit(disp = 0)
#print model.predict(45, 55)