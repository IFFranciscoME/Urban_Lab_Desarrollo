
# .. ................................................................................... .. #
# .. Proyecto: UrbanLab - Plataforma de ayuda para micro y pequeñas empresas             .. #
# .. Archivo: datos.py - procesos de obtencion y almacenamiento de datos                 .. #
# .. Desarrolla: ITERA LABS, SAPI de CV                                                  .. #
# .. Licencia: Todos los derechos reservados                                             .. #
# .. Repositorio: https://github.com/IFFranciscoME/Urban_Lab.git                         .. #
# .. ................................................................................... .. #


# Importar librerias
import pandas as pd
import numpy as np
import geopandas as gpd
import json


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Leer datos y almacenarlos en un DataFrame
# -- ------------------------------------------------------------------------------------ -- #
def read_file(p_file_path, p_sheet):
	"""
	Funcion que recibe en string el path donde se encuentra el archivo xlsx (p_file_path)
	y el nombre de la hoja del excel (p_sheet) que se quiere importar con pandas

    Parameters
    ----------
    p_file_path : str : el nombre del archivo
    p_sheet : str : el nombre de la hoja que se quiere leer

    Returns
    -------
    df_data : DataFrame : dataframe con los datos del excel

    Debugging
    ---------
    p_file_path = 'Base_de_datos.xlsx'
    p_sheet = 'IIEG_E_1'
	
	"""
    # Leer xls
	df_data = pd.read_excel('archivos/' + p_file_path, sheet_name=p_sheet)
	
	return df_data


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Limpiar base de datos de pymes
# -- ------------------------------------------------------------------------------------ -- #
def clean_data_pymes(p_df_data):
	"""
	Funcion que limpia especificamente los datos de las MiPyMEs 
	
    Parameters
    ---------
	p_df_data: DataFrame : datos que se encuentran en un DataFrame

    Returns
    ---------
    df_final: DataFrame : datos limpios

    Debuggin
    ---------
	p_df_data = read_file('Base_de_datos.xlsx', 'IIEG_E_1')

	"""
	# Crear copia
	df = p_df_data.copy()
	
	# Remplazar valores para el procesamientos de los datos
	df.replace([998, 999, 'No contesto', 'No sé'], np.nan, inplace=True)	
	df.replace(
				{
				"Más de 52": 52, 
				"De 26 a 52": 26, 
				"No aplica": 100, 
				"No contesto": 101, 
				"No sé":102, 
				"Más de un año":12}, inplace=True)
	
	# Cambios en columnas especificas
	df['aumento_precios'].replace(100, np.nan, inplace=True)
	
	# Tomar solo la ZMG
	df_final = df.loc[df['Municipio'].isin(['Zapopan','Tonalá',
										  'Tlaquepaque','Tlajomulco de Zúñiga',
										  'El Salto','Guadalajara'])]
	# Reiniciar el index
	df_final.reset_index(drop=True, inplace=True)
	
	return df_final


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Limpiar base de datos de precios
# -- ------------------------------------------------------------------------------------ -- #
def clean_data_prices(p_df_data):
	"""
	Funcion que limpia especificamente los datos de los precios
	
    Parameters
    ---------
    p_df_data: DataFrame : datos que se encuentran en un DataFrame

    Returns
    ---------
    df: DataFrame : datos limpios

    Debuggin
    ---------
	df_data = read_file('Precios_INEGI.xlsx', 'Datos_acomodados')

	"""
	# Hacer copia
	df = p_df_data.copy()
	
	# Quitar numeros innecesarios
	def col_no_numb(df):
		# De las columnas
		col_n = ['División', 'Grupo', 'Clase']
		# Quitar numeros
		no_numb = [df[i].str.replace('\d+', '') for i in col_n]
		# Quitar punto y espacio inicial
		for i in range(len(col_n)):
			point = ['. ', '.. ', '... ']
			df[col_n[i]] = no_numb[i].str.replace(point[i], '', regex=False)
		return df
	df = col_no_numb(df)
	
	# Nombre de todas las columnas con precios acomodadas correctamente
	col_df = list(df.columns)[::-1][0:22]
	
	# Merge ciertas columnas de original con las diferencias
	df_new = pd.merge(
						df[['División', 'Grupo', 'Clase', 'Generico', 'Especificación']], 
						df[col_df].iloc[:,1:], 
				   left_index=True, right_index=True)
	
	return df_new


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Separar series de tiempo del data frame de precios
# -- ------------------------------------------------------------------------------------ -- #
def f_time_series(p_df_prices, p_clase):
	"""
	Funcion que separa las serie de tiempo de acuerdo a la clase que se le pida
	
    Parameters
    ---------
    p_df_prices: DataFrame : data en un DF
	p_clase: str : clase que se requieren los productos

    Returns
    ---------
    series_tiempo: list : todas las series de tiempo

    Debuggin
    ---------
	p_df_prices = read_file('Precios_INEGI.xlsx', 'Datos_acomodados')
	p_clase = 'Accesorios y utensilios'

	"""
	# Agrupar por clase
	clases = list(p_df_prices.groupby('Clase'))
	
	# Busqueda de dataframe para la clase que se necesita
	search = [clases[i][1] for i in range(len(clases)) if clases[i][0]==p_clase][0]
	search.reset_index(inplace=True, drop=True)
	
	# Agrupar por generico
	generico = list(search.groupby('Generico'))
	
	# Series de tiempo por Generico
	series_tiempo = [generico[i][1].median().rename(generico[i][0], 
					     inplace=True) for i in range(len(generico))]
	
	return series_tiempo


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: lista de grupos con cada clase de productos
# -- ------------------------------------------------------------------------------------ -- #
def f_clases(p_df_prices):
	"""
	Funcion que regresa una lista con el nombre de todos los grupos y dentro de la misma
	otra lista con el nombre de todas las clases por grupo
	
    Parameters
    ---------
    p_df_data: DataFrame : data en un DataFrame

    Returns
    ---------
    list_clases: list : todas las clases por grupo

    Debuggin
    ---------
	p_df_prices = read_file('Precios_INEGI.xlsx', 'Datos_acomodados')

	"""
	# Separar por grupo
	group_by_group = list(p_df_prices.groupby('Grupo'))
	
	# lista por grupo y clases por grupo
	list_clases = [[grupo[0], grupo[1]['Clase'].unique().tolist()] for grupo in group_by_group]
	
	return list_clases
		

# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Leer shape file y generar el geojson
# -- ------------------------------------------------------------------------------------ -- #
def read_map_files(p_path_shape, p_path_kml):
	"""
	Funcion que guarda un geojson con los datos para el mapa en 'archivos'
	
    Parameters
    ---------
    p_path_shape: str : path of shape file
	p_path_kml: str : path of kml shape

    Returns
    ---------
    geodf: json : archivo en carpeta de archivos

    Debuggin
    ---------
	 p_path_shape = "archivos/cp_jal_2/CP_14_Jal_v6.shp"
	p_path_kml = "archivos/cp_jal_2/CP_14_Jal_v6.kml"

	"""
	
	# Abrir los archivos y guardalos en DataFrames
	gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
	geodf_kml = gpd.read_file(p_path_kml)
	geodf_shp = gpd.read_file(p_path_shape)
	
	# Convertirlo en GeoJSON
	geodf_kml.to_file(p_path_kml, driver="GeoJSON")

	with open(p_path_kml) as geofile:
		j_file = json.load(geofile)
	# Asignar el id al kml
	i = 0
	for feature in j_file["features"]:
		feature['id'] = geodf_shp['d_cp'][i]
		i += 1
	
	# Guardalo en los archivos
	with open('archivos/CP.json', 'w') as fp:
		   json.dump(j_file, fp)
