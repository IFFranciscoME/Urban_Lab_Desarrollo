# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 00:27:24 2020

@author: sally
"""

import pandas as pd
import numpy as np
import math

# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Calcular metrica con diccionario
# -- ------------------------------------------------------------------------------------ -- #
def metric_quantification(p_df_data, p_conditions, p_metric):
	"""
	Funcion que pasa un diccionario con condiciones por columna al dataframe de pymes
	que regresa la suma de la metrica de acuerdo a las condiciones que se le de
	
    Parameters
    ---------
    p_df_data: DataFrame : datos de pymes en Dataframe
	p_conditions: dict : diccionario con condiciones
	p_metric: str : nombre de la metrica

    Returns
    ---------
    df: dict : Diccionario con metrica en el df original y como matriz

    Debuggin
    ---------
	p_df_data = datos.clean_data_pymes((datos.read_file(
							'Base_de_datos.xlsx', 'IIEG_E_1'))
	p_conditions = entradas.conditions_stress
	p_metric = 'Estres'
	
	"""
	# Nombre de columnas
	list_columns = list(p_conditions.keys())
	# Condiciones (dicts)
	list_dict_conditions = list(p_conditions.values())
	# Lista de lista con resultados
	answer = [[f_condition(
							list_dict_conditions[k], 
							   p_df_data[list_columns[k]][i]
							   ) 
							for i in range(len(p_df_data))
							] 
					for k in range(len(list_columns))
					]
	# DataFrame con la matriz de todo
	metric = pd.DataFrame(answer)
	
	# --------------------------
	# Columna con suma
	metric_sum = metric.sum()
	
	# Nombre de variables para columnas
	col = list(p_conditions.keys())
	
	# Transponer tabla de metrica
	metric_table = metric.T
	
	# Asignar nombres a las columnas
	metric_table.columns = col
	
	# Agregar columna de suma total
	metric_table['Total'] = metric_sum
	
	# --------------------------
	
	# Dataframe copia
	df = p_df_data.copy()
	
	# Dataframe con columna de metrica
	df[p_metric] = metric_sum
	
	return {'df_prices': df, 'metric_table': metric_table}


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Regresa el resultado de un dato de acuerdo a las condiciones
# -- ------------------------------------------------------------------------------------ -- #
def f_condition(p_dict_condition, p_data):
	"""
	Funcion que checa en el diccionario de condiciones para las metricas
	el diccionario de condiciones contiene dentro otros diccionarios, donde:
		el value del dict dentro del dict es p_codition (la condicion que se debe de cumplir)
		y el key es el resultado que se asigna si cumple la condicion
	esto lo checa para cada dato de acuerdo a la columna (key del dict) en otra funcion
	
    Parameters
    ---------
    p_dict_condition: dict : diccionario con condiciones
	p_data: int or str: dato a comparar
    
	Returns
    ---------
    int: valor de acuerdo a la condicion

    Debuggin
    ---------
	p_dict_condition = list(entradas.conditions_stress.values())[0]
	p_data = datos.clean_data_pymes((datos.read_file(
							'Base_de_datos.xlsx', 'IIEG_E_1'))['ventas_porcentaje'][0]
	
	"""
	# Valores que se podrian poner
	posible_results = list(p_dict_condition.keys())
	
	# lista de condiciones
	list_conditions = list(p_dict_condition.values())
	
	# Utilizando la funcion para cada condicion
	answer = [f_type_verification(list_conditions[i], posible_results[i], 
							p_data) for i in range(len(list_conditions ))]
	
	# Si todos son cero, quiere decir que p_data es nan
	if answer == [0]*len(answer):
		return 0
	
	# Si no
	else:
		# Quitar los nan de la lista answer
		lista = list(filter(None.__ne__, answer))
		if len(lista) == 0:
			return np.nan
		else:
			# Regresar el resultado
			return lista[0]
		

# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Checar la condicion de acuerdo a su tipo
# -- ------------------------------------------------------------------------------------ -- #	
def f_type_verification(p_condition, p_result, p_data):
	"""
	Funcion que de acuerdo a los 3 parametros te regresa un resultado, 
		p_data puede ser string o int, pero si p_condition es una lista se busca en tal
		si esta se devuelve p_result, si p_data no es string entonces es numerico
		si es nan devuelve 0 si es tupla, debe de estar entre ambos numeros de la tupla
	
    Parameters
    ---------
    p_condition: tuple or list : contiene las condiciones
	p_result: int : numero si se cumple la condicion es el resultado
	p_data: int or str: dato que se esta comparando para cuantificar
		
    Returns
    ---------
    p_result: int : numero de la metrica

    Debuggin
    ---------
	p_condition = (0, 25)
	p_result = 1
	p_data = 10
		
	"""
	# Si es lista tiene que estar en tal
	if type(p_condition) == list:
		if p_data in p_condition:
			return p_result
	
	# Si es numerico
	if type(p_data) != str:
		if math.isnan(p_data):
			return 0
		# Si es tuple, tiene que estar entre ambos numeros
		if type(p_condition) == tuple:
			if p_condition[0] < p_data and p_data <= p_condition[1]:
				return p_result
