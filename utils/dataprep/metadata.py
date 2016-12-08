from .json_to_csv import *
from .musescore_api.MuseScoreAPI import MuseScoreAPI
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

client_key=config.get('DEFAULT', 'musescore_api_key')
api = MuseScoreAPI(client_key=client_key)