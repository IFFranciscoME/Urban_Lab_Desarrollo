
# .. ................................................................................... .. #
# .. Proyecto: SocialSim - Plataforma de simulacion de proyectos socioproductivos        .. #
# .. Archivo: visualizaciones.py - diccionario con datos de entrada                      .. #
# .. Desarrolla: ITERA LABS, SAPI de CV                                                  .. #
# .. Licencia: Todos los derechos reservados                                             .. #
# .. Repositorio: https://github.com/IFFranciscoME/Urban_Lab.git                         .. #
# .. ................................................................................... .. #

# Importing and initializing main Python libraries
import plotly.express as px
import geojson

# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Map
# -- ------------------------------------------------------------------------------------ -- #
def map_metric(df_data, metric):
	"""
    Parameters
    ---------
    :param:
        df_data: DataFrame : data in a DF
		metric: str : name of column of metric
		path: str : "cp_jal_2/CP_14_Jal_v6.shp"

    Returns
    ---------
    :return:
        figure

    Debuggin
    ---------
        df_data = metric_quantification(df_data, ent.conditions_stress, 'Estres')
		metric = 'estres'
		path: str : "cp_jal_2/CP_14_Jal_v6.shp"
	"""
	with open('archivos/CP.json') as f:
		   j_file = geojson.load(f)
	fig = px.choropleth_mapbox(df_data, geojson=j_file, locations= 'CP', color=metric,
	                           color_continuous_scale="Viridis",
	                           range_color=(0, max(df_data[metric])),
	                           mapbox_style="carto-positron",
	                           zoom=10, center = {"lat": 20.666820, "lon": -103.3918228},
	                           opacity=0.5,
	                           labels={metric:'Nivel de ' + metric}
	                          )
	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	return fig




