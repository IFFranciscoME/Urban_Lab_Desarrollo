
# .. ................................................................................... .. #
# .. Proyecto: UrbanLab - Plataforma de ayuda para micro y pequeñas empresas             .. #
# .. Archivo: principal.py - flujo principal de uso                                      .. #
# .. Desarrolla: ITERA LABS, SAPI de CV                                                  .. #
# .. Licencia: Todos los derechos reservados                                             .. #
# .. Repositorio: https://github.com/IFFranciscoME/Urban_Lab.git                         .. #
# .. ................................................................................... .. #


# Importing and initializing main Python libraries
import proceso as pr
import datos as dat
import entradas as ent
import visualizaciones as vs
from time import time
import warnings
warnings.filterwarnings('ignore')

# Start time
#t0 = time()

if __name__ == "__main__":
	
	# Leer datos (orig) de mipymes
	df_pymes_or = dat.read_file(ent.path_data_pyme, ent.sheet_data_pyme)
	
	# Limpiar datos de mipymes
	df_pymes = dat.clean_data_pymes(df_pymes_or)
	
	# Usar funcion de cuantificar 
	metric_s = pr.metric_quantification(df_pymes, ent.conditions_stress, 'Estres')
	
	# Dataframe de metrica de estres
	df_stress = metric_s['df_prices']
	# tabla de metrica
	metric_s_table = metric_s['metric_table']
	
	# Using metric_quantification with adaptability conditions
	metric_a = pr.metric_quantification(df_pymes, ent.conditions_adaptability, 'Adaptabilidad')
	
	# Dataframe de metrica de estres
	df_adapt = metric_a['df_prices']
	# tabla de metrica
	metric_a_table = metric_a['metric_table']
	
	# -- VISUALIZACIONES -- #
	
	# Mapa
	
	#map_s  = vs.map_metric(df_stress, 'Estres', ent.dict_colors['Estres'])
	#map_a = vs.map_metric(metric_a, 'Adaptabilidad', ent.dict_colors['Adaptabilidad'])
	
	
	# Velocimetro
	
	#vel_s = vs.velocimeter_size(df_pymes, 'Estres', metric_s_table)
	#vel_a = vs.velocimeter_size(df_pymes, 'Adaptabilidad', metric_a_table)
	
	# Barras
	
	#b_a = vs.bars_city(df_pymes, 'Estres', metric_s_table)
	#b_a = vs.bars_city(df_pymes, 'Adaptabilidad', metric_a_table)
	
	# Tabla
	#t_s = vs.table_giro(df_pymes, 'Estres', metric_s_table)
	#t_a = vs.table_giro(df_pymes, 'Adaptabilidad', metric_a_table)
	
	# .. ............................................................................... .. #
	# .. ............................................................................... .. #
	
	# Leer base de datos de precios original
	df_prices_or = dat.read_file(ent.path_data_prices, ent.sheet_data_prices)
	
	# limpiar base de datos
	df_prices = dat.clean_data_prices(df_prices_or)
	
	# Semaforo
	semaforo = pr.semaforo_precios(df_prices)
	
	# Dataframe multi index
	predicciones = semaforo['predicciones']
	
	 
	# -- VISUALIZACIONES -- #
	
	# Tabla de semaforo
	#tab_p = vs.table_prices(semaforo['semaforo'])
	
	# Comparacion de precios
	#dif_p = vs.dif_prices(predicciones, 'alimentos')
	
	# End time
	#t1 = time()
	
	#print('el tiempo transcurrido fue: ' + str(t1-t0))
	

	
	
	
	
	
		
		
	
	
	
	
	