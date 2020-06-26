
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
import matplotlib.pyplot as plt

# Start time
#t0 = time()

if __name__ == "__main__":
	
	# Using function: read_file (original)
	df_pymes_or = dat.read_file(ent.path_data_pyme, ent.sheet_data_pyme)
	
	# Using function: clean_data
	df_pymes = dat.clean_data_pymes(df_pymes_or)
	"""
	# Using metric_quantification with stress conditions
	metric_s = pr.metric_quantification(df_pymes, ent.conditions_stress, 'Estres')
	
	# Using metric_quantification with adaptability conditions
	metric_a = pr.metric_quantification(df_pymes, ent.conditions_adaptability, 'Adaptabilidad')
	
	# Visualizations
	fig  = vs.map_metric(metric_s, 'Estres')
	fig2 = vs.map_metric(metric_a, 'Adaptabilidad')
	
	'''
	#fig.show()
	#fig2.show()
	
	'''
	"""
	# .. ............................................................................... .. #
	# .. ............................................................................... .. #
	
	# Leer base de datos de precios original
	df_prices_or = dat.read_file(ent.path_data_prices, ent.sheet_data_prices)
	
	# limpiar base de datos
	df_prices = dat.clean_data_prices(df_prices_or)
	#%%
	# Fragmentar por series de tiemo
	time_series = dat.series_tiempo(df_prices)
	#%%
	#plot series
	ts = 1
	#arima = pr.fit_arima(time_series[ts])
	arimas = [pr.fit_arima(s) for s in time_series]
	time_series[ts].plot()
	#%%
	# Todas las arimas
	#arimas_f = pr.all_arimas(df_prices)
	
	#%%
	"""
	st = 12
	result = pr.forecast(arimas_f[st][1], time_series[st])
	
	#import matplotlib.pyplot as plt
	plt.plot(time_series[st])
	#arimas_f[st][1].plot_predict(dynamic=False)
	plt.plot(result[-7:])
	plt.xticks(rotation=90)
	plt.show()
	"""
	
	'''
	# End time
	#t1 = time()
	
	#print('el tiempo transcurrido fue: ' + str(t1-t0))
	'''
	
	
	
	
	
		
		
	
	
	
	
	