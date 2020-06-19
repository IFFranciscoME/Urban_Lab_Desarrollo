
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

# Start time
t0 = time()

if __name__ == "__main__":
	
	# Using function: read_file (original)
	df_data_or = dat.read_file(ent.data_path, ent.data_sheet)
	
	# Using function: clean_data
	df_data = dat.clean_data(df_data_or)
	
	# Using metric_quantification with stress conditions
	metric_s = pr.metric_quantification(df_data, ent.conditions_stress, 'Estres')
	
	# Using metric_quantification with adaptability conditions
	metric_a = pr.metric_quantification(df_data, ent.conditions_adaptability, 'Adaptabilidad')
	
	# End time
	t1 = time()
	
	# Visualizations
	fig = vs.map_metric(metric_s, 'Estres', ent.shp_path, ent.kml_path)
	fig.show()
	#vs.map_metric(metric_a, 'Adaptabilidad', ent.shp_path, ent.kml_path)

	#print('el tiempo transcurrido fue: ' + str(t1-t0))