
# .. ................................................................................... .. #
# .. Proyecto: UrbanLab - Plataforma de ayuda para micro y pequeñas empresas             .. #
# .. Archivo: datos.py - procesos de obtencion y almacenamiento de datos                 .. #
# .. Desarrolla: ITERA LABS, SAPI de CV                                                  .. #
# .. Licencia: Todos los derechos reservados                                             .. #
# .. Repositorio: https://github.com/IFFranciscoME/Urban_Lab.git                         .. #
# .. ................................................................................... .. #


# Importing and initializing main Python libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import json

# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Read data file and storing it in to a DataFrame
# -- ------------------------------------------------------------------------------------ -- #
def read_file(file_path, sheet):
    """
    Parameters
    ---------
    :param:
        file: str : name of file xlsx
		sheet: str : name of sheet
    Returns
    ---------
    :return:
        df_data: DataFrame : file's data

    Debuggin
    ---------
        file_path = 'Base_de_datos.xlsx'
		sheet = 'IIEG_E_1'

    """
    # Read xls
    df_data = pd.read_excel('archivos/' + file_path, sheet_name=sheet)
    return df_data


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Cleaning Database that is in a DataFrame
# -- ------------------------------------------------------------------------------------ -- #
def clean_data_pymes(df_data):
	"""
    Parameters
    ---------
    :param:
        df_data: DataFrame : data in a DF

    Returns
    ---------
    :return:
        df: DataFrame : clean data in DF

    Debuggin
    ---------
        df_data = read_file(ent.path, ent.sheet)

	"""
	# Make a copy
	df = df_data.copy()
	# Replace
	df.replace([998, 999, 'No contesto', 'No sé'], np.nan, inplace=True)	
	df.replace(
				{
				"Más de 52": 52, 
				"De 26 a 52": 26, 
				"No aplica": 100, 
				"No contesto": 101, 
				"No sé":102, 
				"Más de un año":12}, inplace=True)
	
	# Specific columns
	df['aumento_precios'].replace(100, np.nan, inplace=True)

	return df


# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Cleaning Database that is in a DataFrame
# -- ------------------------------------------------------------------------------------ -- #
def clean_data_prices(df_data):
	"""
    Parameters
    ---------
    :param:
        df_data: DataFrame : data in a DF

    Returns
    ---------
    :return:
        df: DataFrame : clean data in DF

    Debuggin
    ---------
        df_data = read_file(ent.path, ent.sheet)

	"""
	# Hacer copia
	df = df_data.copy()
	
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
# -- Function: Separar series de tiempo del data frame de series de tiempo
# -- ------------------------------------------------------------------------------------ -- #
def series_tiempo(df_data, p_clase):
	"""
    Parameters
    ---------
    :param:
        df_data: DataFrame : data en un DF
		p_clase: str : clase que se requieren los productos

    Returns
    ---------
    :return:
        series_tiempo: list : todas las series de tiempo

    Debuggin
    ---------
        df_data = df_prices
		p_clase = 'Accesorios y utensilios'

	"""
	# Agrupar por clase
	clases = list(df_data.groupby('Clase'))
	
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
def clases(df_prices):
	"""
    Parameters
    ---------
    :param:
        df_data: DataFrame : data en un DF
		p_clase: str : clase que se requieren los productos

    Returns
    ---------
    :return:
        series_tiempo: list : todas las series de tiempo de productos de la clase

    Debuggin
    ---------
        df_data = df_prices
		p_clase = 'Accesorios y utensilios'

	"""
	# Separar por grupo
	group_by_group = list(df_prices.groupby('Grupo'))
	# lista por grupo y clases por grupo
	list_clases = [[grupo[0], grupo[1]['Clase'].unique().tolist()] for grupo in group_by_group]
	return list_clases
		

# -- ------------------------------------------------------------------------------------ -- #
# -- Function: Read shape file and storing it in to a DataFrame
# -- ------------------------------------------------------------------------------------ -- #
def read_map_files():
	"""
    Parameters
    ---------
    :param:
        path: str : path of shape file

    Returns
    ---------
    :return:
        geodf: DataFrame : clean data in DF

    Debuggin
    ---------
        path = ent.map_path

	"""
	# Donde se encuentran los archivos
	path_shape = "archivos/cp_jal_2/CP_14_Jal_v6.shp"
	path_kml = "archivos/cp_jal_2/CP_14_Jal_v6.kml"
	
	# Abrir los archivos y guardalos en DataFrames
	gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
	geodf_kml = gpd.read_file(path_kml)
	geodf_shp = gpd.read_file(path_shape)
	
	# Convertirlo en GeoJSON
	geodf_kml.to_file(path_kml, driver="GeoJSON")

	with open(path_kml) as geofile:
		j_file = json.load(geofile)
	# Asignar el id al kml
	i = 0
	for feature in j_file["features"]:
		feature['id'] = geodf_shp['d_cp'][i]
		i += 1
	
	# Guardalo en los archivos
	with open('archivos/CP.json', 'w') as fp:
		   json.dump(j_file, fp)
