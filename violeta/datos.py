
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
def clean_data(df_data):
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
