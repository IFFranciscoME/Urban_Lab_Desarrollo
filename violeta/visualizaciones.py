
# .. ................................................................................... .. #
# .. Proyecto: SocialSim - Plataforma de simulacion de proyectos socioproductivos        .. #
# .. Archivo: visualizaciones.py - diccionario con datos de entrada                      .. #
# .. Desarrolla: ITERA LABS, SAPI de CV                                                  .. #
# .. Licencia: Todos los derechos reservados                                             .. #
# .. Repositorio: https://github.com/IFFranciscoME/Urban_Lab.git                         .. #
# .. ................................................................................... .. #


# Importing and initializing main Python libraries
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, HoverTool
from bokeh.palettes import brewer
import datos as dat


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Map
# -- ------------------------------------------------------------------------------------ -- #
def map_metric(df_data, metric, path):
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
	# Shape file data
	geodf = dat.read_map_file(path)
	
	# GeoJSON with metric data
	json_data = dat.merge_data(df_data, geodf, metric)
	
	# Input GeoJSON source that contains features for plotting.
	geosource = GeoJSONDataSource(geojson = json_data)
	# Define a sequential multi-hue color palette.
	palette = brewer['YlGnBu'][8]
	# Reverse color order so that dark blue is highest obesity.
	palette = palette[::-1]
	# Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
	color_mapper = LinearColorMapper(palette = palette, low = 0, high = 40, nan_color = '#d9d9d9')
	
	# Add hover tool
	hover = HoverTool(tooltips = [ ('CP:','@d_cp'),(metric + ':', '@'+metric)])
	
	# Define custom tick labels for color bar.
	tick_labels = {'0': '0', '5': '5', '10':'10', '15':'15', '20':'20',
	               '25':'25', '30':'30','35':'35', '40': '>40'}
	# Create color bar. 
	color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
	border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)
	# Create figure object.
	p = figure(title = metric, plot_height = 600 ,
	           plot_width = 950, toolbar_location = 'below',
	           tools = [hover, "zoom_in", "zoom_out", "reset", "pan"],
	           x_range = (2330000, 2380000), y_range = (940000, 980000))
	p.xgrid.grid_line_color = None
	p.ygrid.grid_line_color = None
	
	# Add patch renderer to figure. 
	p.patches('xs','ys', source = geosource,fill_color = {'field' :metric, 'transform' : color_mapper},
	          line_color = 'black', line_width = 0.25, fill_alpha = 1)
	# Specify figure layout.
	
	p.add_layout(color_bar, 'below')
	
	# Display figure inline in Jupyter Notebook.
	#output_notebook()
	
	# Display figure.
	show(p)
	
# map_metric(pr.metric_s, 'estres', ent.map_path)




