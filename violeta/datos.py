
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
						df[col_df].pct_change(axis=1).iloc[:,1:], 
				   left_index=True, right_index=True)
	return df_new

def series_tiempo(df_data):
	# Agrupar por clase
	clases = list(df_data.groupby('Clase'))
	# Series de tiempo con promedio de la clase
	series_tiempo_or = [clases[i][1].mean().rename(clases[i][0], 
					     inplace=True) for i in range(len(clases))]
	
	series_tiempo = [np.asarray(st.reset_index(drop=True)) for st in series_tiempo_or]
	# del series_tiempo[7]
	return series_tiempo
		

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
