
# .. ................................................................................... .. #
# .. Proyecto: SocialSim - Plataforma de simulacion de proyectos socioproductivos        .. #
# .. Archivo: visualizaciones.py - diccionario con datos de entrada                      .. #
# .. Desarrolla: ITERA LABS, SAPI de CV                                                  .. #
# .. Licencia: Todos los derechos reservados                                             .. #
# .. Repositorio: https://github.com/IFFranciscoME/Urban_Lab.git                         .. #
# .. ................................................................................... .. #


# Importar librerias
import plotly.express as px
import plotly.graph_objects as go
import geojson
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Mapa
# -- ------------------------------------------------------------------------------------ -- #
def map_metric(p_df_data, metric, color):
	"""
	Visualizacion de mapa por metrica
	
    Parameters
    ---------
    p_df_data: DataFrame : datos de pymes con columna de metrica
	metric: str : nombre de la metrica
	color: str : colores para metrica

    Returns
    ---------
    fig : plotly figure : mapa de la zmg de jalisco

    Debuggin
    ---------
	p_df_data = metric_quantification(df_pymes, ent.conditions_stress, 
								   'Estres')['df_prices']
	metric = 'Estres'
	color = "Reds"
	
	"""
	with open('archivos/CP.json') as f:
		   j_file = geojson.load(f)
	fig = px.choropleth_mapbox(p_df_data, geojson=j_file, locations= 'CP', color=metric,
	                           color_continuous_scale=color,
	                           range_color=(min(p_df_data[metric]), max(p_df_data[metric])),
	                           mapbox_style="carto-positron",
	                           zoom=10, center = {"lat": 20.666820, "lon": -103.3918228},
	                           opacity=0.5,
	                           labels={metric:'Nivel de ' + metric}
	                          )
	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	return fig


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Velocimetro por tamaño de pyme
# -- ------------------------------------------------------------------------------------ -- #
def velocimeter_size(p_df_data, p_metric, p_metric_table):
	"""
    Parameters
    ---------
    p_df_data: DataFrame : datos de pymes en dataframe
	p_metric: str : nombre de la metrica
	p_metric_table: DataFrame : matriz del calculo de la metrica

    Returns
    ---------
    fig : plotly figure : mapa de la zmg de jalisco

    Debuggin
    ---------
	p_df_data = df_pymes
	p_metric = 'Estres'
	p_metric_table = pr.metric_quantification(df_pymes, 
									   ent.conditions_stress, 'Estres')['metric_table']
	
	"""
	# Añadir columna necesaria
	p_metric_table['Tamaño'] = p_df_data['Tamaño']
	# Creación de tabla resumen
	pivot_size = pd.pivot_table(p_metric_table,index = ['Tamaño'],
                                values = list(p_metric_table.columns), aggfunc=np.mean) 
	# Creacion de figura
	fig = go.Figure()
	
	# limites
	lim = p_metric_table['Total'].max()
	
	fig.add_trace(go.Indicator(
        value = pivot_size.loc['Micro','Total'],
        title = {'text': "Micro"},
        gauge = {
            'bar': {'color': "mediumpurple"},
            'axis': {'range': [None, lim],'visible': False}},
        domain = {'row': 0, 'column': 0}))
	
	fig.add_trace(go.Indicator(
        value = pivot_size.loc['Pequeña','Total'],
        title = {'text': "Pequeña"},
        gauge = {
            'bar': {'color': "purple"},
            'axis': {'range': [None, lim],'visible': False}},
        domain = {'row': 0, 'column': 1}))
	
	fig.add_trace(go.Indicator(
        value = pivot_size.loc['Mediana','Total'],
        title = {'text': "Mediana"},
        gauge = {
            'bar': {'color': "purple"},
            'axis': {'range': [None, lim],'visible': False}},
        domain = {'row': 1, 'column': 0}))
	
	fig.add_trace(go.Indicator(
        value = pivot_size.loc['Grande','Total'],
        title = {'text': "Grande"},
        gauge = {
            'bar': {'color': "purple"},
            'axis': {'range': [None, lim],'visible': False}},
        domain = {'row': 1, 'column': 1}))
	
	fig.update_layout(title= p_metric + " en los Tamaños de las empresas",
                      grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
                      template = {'data' : {'indicator': [{
                          'title': {'text': "Tamaño"},
                          'mode' : "number+gauge"}]}})
	return fig


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: De barras por municipio
# -- ------------------------------------------------------------------------------------ -- #
def bars_city(p_df_data, p_metric, p_metric_table):
	"""
    Parameters
    ---------
    p_df_data: DataFrame : datos de pymes en dataframe
	p_metric: str : nombre de la metrica
	p_metric_table: DataFrame : matriz del calculo de la metrica

    Returns
    ---------
    fig : plotly figure : mapa de la zmg de jalisco

    Debuggin
    ---------
	p_df_data = df_pymes
	p_metric = 'Estres'
	p_metric_table = pr.metric_quantification(df_pymes, 
									   ent.conditions_stress, 'Estres')['metric_table']
	
	"""
    # Añadir columna necesaria
	p_metric_table['Municipio'] = p_df_data['Municipio']
    # Creación de tabla resumen
	pivot_mun = pd.pivot_table(p_metric_table,index = ['Municipio'],
                                values = list(p_metric_table.columns), aggfunc=np.mean) 
    # Creacion de figura
	fig = go.Figure()
    # Colores
	colors = ['darkviolet', 'mediumpurple', 'mediumpurple', 'thistle', 'thistle','mediumpurple']
    # limites
	lim = p_metric_table['Total'].max()
    # figura
	fig = go.Figure(data=[go.Bar(x=pivot_mun.index, y=pivot_mun['Total'],marker_color=colors)],
               layout={'yaxis':{'range': [0, lim]}})
	fig.update_layout(title= p_metric + " en los Municipios de la ZMG",
                  xaxis_title="Municipios",
                  yaxis_title=p_metric,
                  plot_bgcolor='white',hovermode='closest')
	
	return fig


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Tabla de giro
# -- ------------------------------------------------------------------------------------ -- #
def table_giro(p_df_data, p_metric, p_metric_table):
	"""
    Parameters
    ---------
    p_df_data: DataFrame : datos de pymes en dataframe
	p_metric: str : nombre de la metrica
	p_metric_table: DataFrame : matriz del calculo de la metrica

    Returns
    ---------
    fig : plotly figure : mapa de la zmg de jalisco

    Debuggin
    ---------
	p_df_data = df_pymes
	p_metric = 'Estres'
	p_metric_table = pr.metric_quantification(df_pymes, 
									   ent.conditions_stress, 'Estres')['metric_table']
	
	
	"""
	# Agregar columna de giro
	p_metric_table['Giro'] = p_df_data['Giro']
	# Creación de tabla resumen
	pivot_giro = pd.pivot_table(p_metric_table,index = ['Giro'],
	                                values = list(p_metric_table.columns), aggfunc=np.median) 
	# Cambiar nombre de columnas
	change_name_columns = lambda x: x.replace('_', ' ').title()
	# Nuevas columnas
	new_col = list(map(change_name_columns, list(pivot_giro.columns)))[1:]
	
	# En vez de total, nombre de metrica
	new_col.insert(0, p_metric)
	
	# Renombrar columnas de la pivot de giro
	pivot_giro.columns = new_col
	
	# Columnas de giro
	column_giro = list(pivot_giro.columns)
	
	# Tomar lista de columnas de la pivote
	list_col = [pivot_giro[c] for c in column_giro]
	
	# Agregar el indice
	list_col.insert(0, list(pivot_giro.index))
	
	fig = go.Figure(data=[go.Table(name=p_metric,
	header=dict(values= ["Giro"]+pivot_giro.columns.tolist(),
	                fill_color='mediumpurple',
	                align='center', font_color = 'White', font_size = 11),
	    cells=dict(values=list_col,
	               fill_color='thistle',
	               align='center',height=40, font_size=11))])
	
	return fig

	
# -- ------------------------------------------------------------------------------------ -- #
# -- Function: tabla de precios
# -- ------------------------------------------------------------------------------------ -- #
def table_prices(p_df_semaforo):
	"""
	Visualizacion de la tabla en plotly del semaforo de precios que se calcula con
	la funcion de semaforo que se encuentra en proceso
	
    Parameters
    ---------
    p_df_semaforo: DataFrame : datos del semaforo en un DataFrame

    Returns
    ---------
    fig : plotly figure : tabla de semforo de precios

    Debuggin
    ---------
	p_df_semaforo = proceso.semaforo_precios(df_prices)['semaforo']

	
	"""
    # resultados
	values = list(p_df_semaforo[0])
    # colores verde | amarillo | rojo
	colores = ['rgb(230, 120, 100)', 'rgb(245, 220, 130)', 'rgb(200, 230, 150)']
    # asignar colores
	asig_colors = lambda x: colores[2] if x=='rojo' else (colores[0] if x=='verde' else colores[1])
    # colores de tabla
	colors_t = list(map(asig_colors, values))
    
	fig = go.Figure(data=[go.Table(
      header=dict(
        values=['<b>Productos - grupo</b>', '<b>Resultado de predicción %</b>'],
        line_color='white', fill_color='rgb(240, 225, 230)',
        align='center',font=dict(family="Old Standard TT", color='rgb(110, 90, 100)', size=18)
      ),
      cells=dict(
        values=[list(p_df_semaforo.index), list(p_df_semaforo[1])],
        line_color=['rgb(235, 225, 245)', 'white'],
        fill_color=['white', colors_t],
        align='center', font=dict(family="Old Standard TT",color='rgb(130, 120, 120)', size=12)
        ))
    ])
	
	fig.update_layout(height=400, margin=dict(r=7, l=4, t=6, b=6))
	
	return fig


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: diferencias de los precios
# -- ------------------------------------------------------------------------------------ -- #
def dif_prices(p_df_predicciones, p_grupo):
	"""
	Visualizacion de las diferencias de la mediana del ultimo precio por clase
	con respecto a la prediccion
	
    Parameters
    ---------
    p_df_predicciones: DataFrame : resultado de predicciones
	p_grupo: str : nombre del grupo de productos

    Returns
    ---------
    fig : plotly figure : mapa de la zmg de jalisco

    Debuggin
    ---------
	p_df_predicciones = proceso.semaforo_precios(df_prices)['predicciones']
	p_grupo = 'Alimentos'
	
	"""
    # Transponer 
	predicciones = p_df_predicciones.T
    # Tomar datos del grupo
	grupo = predicciones[p_grupo]
    
	if len(grupo.T)==8:
		# Tipo de fig que se añadiran al subplot
		tipos = [[{"type": "indicator"}, {"type": "indicator"}] for i in range(4)]
		fig = make_subplots(rows=4, cols = 2, specs=tipos, 
					  vertical_spacing=0.2, horizontal_spacing=0.25)
		rows = [1,2,3,4,1,2,3,4]
		cols = [1,1,1,1,2,2,2,2]
        
		# Para las 8 
		for i in range(len(grupo.T)):
			
            # Datos que se necesitan de cada clase del grupo
			titulo = list(grupo.columns)[i]
			ultimo_precio = grupo.iloc[0,i]
			futuro_precio = grupo.iloc[1,i]
            
			# Añadir fig
			fig.append_trace( go.Indicator(
                mode = "number+gauge+delta", value = futuro_precio,
                delta = {'reference': ultimo_precio},
                title = {'text' : titulo, 'font': {'size': 9}},
                gauge = {
                    'shape': "bullet",
                    'axis': {'range': [ultimo_precio*.85, futuro_precio*1.1]},
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': ultimo_precio},
                    'steps': [
                        {'range': [ultimo_precio*.85, ultimo_precio], 'color': "lightgray"}],
                    'bar': {'color': "black"}}), row=rows[i], col=cols[i])
		# Layout de fig
		fig.update_layout(title = p_grupo, title_x=0.5, 
							height = 450, margin = {'t':100, 'b':10, 'l':150},
							font = {'color': "darkblue", 'family': "Arial", 'size': 14}
                         )
    
	else:
        # Los tipos de graficas
		tipos = [[{"type": "indicator"}] for i in range(len(grupo.T))]

        # Hacer subplots
		fig = make_subplots(rows=len(grupo.T), cols = 1, specs=tipos, vertical_spacing=0.2)
		
		for i in range(len(grupo.T)):
			
			# Datos que se necesitan de cada clase del grupo
			titulo = list(grupo.columns)[i]
			ultimo_precio = grupo.iloc[0,i]
			futuro_precio = grupo.iloc[1,i]
			
			# Añadir plots
			fig.append_trace( go.Indicator(
                mode = "number+gauge+delta", value = futuro_precio,
                delta = {'reference': ultimo_precio},
                domain = {'x': [0.25, 1], 'y': [0.7, 0.9]},
                title = {'text' : titulo, 'font': {'size': 11}},
                gauge = {
                    'shape': "bullet",
                    'axis': {'range': [ultimo_precio*.85, futuro_precio*1.1]},
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': ultimo_precio},
                    'steps': [
                        {'range': [ultimo_precio*.85, ultimo_precio], 'color': "lightgray"}],
                    'bar': {'color': "black"}}), row=i+1, col=1)
		
		# Layout general
		fig.update_layout(title = p_grupo, title_x=0.5, 
							height = 450, margin = {'t':100, 'b':10, 'l':250, 'r': 10},
							font = {'color': "darkblue", 'family': "Arial", 'size': 14})
		
	return fig




