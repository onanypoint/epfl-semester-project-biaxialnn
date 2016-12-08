from .json_to_csv import *
from .musescore_api.MuseScoreAPI import MuseScoreAPI
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

client_key=config.get('DEFAULT', 'musescore_api_key')
api = MuseScoreAPI(client_key=client_key)

def get_page(page, params):
    """Query musescore api
    
    Parameters
    ----------
    page : int
        the page number to retrieve
    params : dict
        parameteras used for the api call
    
    Returns
    -------
    MuseScoreAPI.MuseScoreResponse
        The query response
    """
    params.update({'page':page})
    return api.request(resource='score', params=params)

def get_default_params(part=None, parts=None, sort='relevance'):
    """Get query parameters
    
    Parameters
    ----------
    part : int
        [-1;-128] a midi program number, zero based indexed. Drumset is 128 
        and -1 is undefined (the default is None, which does retrieve. 
        every part)
    parts : int
        The number of parts in the score. 1 for solo scores. 2 for duo etc…
        (the default is None, which does not impose a number of part).
    sort : str, optional
        How to sort the scores.
    
    Returns
    -------
    dict
        Dictionary containing the different parameters
    """
    if part : assert parts 

    params = {}  
    if part: params.update({'part':part})
    if parts: params.update({'parts':parts})
    if sort: params.update({'sort':sort})   
    return params

def write_batch(batch, file, write_key=False):
    """Write a json to a csv file
    
    Parameters
    ----------
    batch : json
        The json two write to csv
    file : csv file writer
        Where to write the csv data
    """
    keys, rows = dicts_to_csv(json_to_dicts(batch))
    if write_key:
        write_csv(file, keys=keys, rows=rows)
    else:
        write_csv(file, rows=rows)
    
def get_metadata(directory, prefix, part=None, parts=None, retrieve_max=100000):
  """Retrieve musescore score metadata
    
    This method retreive the musescore score metadata based on the api result.
    It writes directly the retrieved data to disk in a "unrolled" csv where 
    the json data is flattened.
    
    Parameters
    ----------
    directory : PostfixPath
        The path to where the data should be saved
    prefix : str
        Prefix to use for the file
    part : int, optional
    parts : int, optional
    retrieve_max : int, optional
        maximum number of metadata to retrieve (the default is 100000)
    """
    SCORE_PER_PAGE = 20
    WRITE_THRESHOLD = 500
    
    retrieve_max = retrieve_max - retrieve_max%SCORE_PER_PAGE
    file = directory/(prefix + '_meta.csv')

    if file.is_file():
        print('File already exists, aborting.')
        return file

    with open(str(file), 'w') as output_file:
        cw = csv.writer(output_file)
        default_params = get_default_params(part, parts)
        
        print("Requesting...")
        print("At most",retrieve_max,"scores metadata are going to be retrieved")
        
        def write(results, write_key=False):
            write_batch(json.dumps(results), cw, write_key)  
            output_file.flush()
            print('Batch written (iteration {})'.format(i)) 
            return []

        results = [] 
        for i in range(int(retrieve_max/SCORE_PER_PAGE)):    
            r = get_page(i, default_params)
            data = json.loads(r.text)
            
            if not data:
                print('No more data to fetch.')
                break
            
            results.extend(data)  
            
            if i == 0:
                results = write(results, write_key=True)
            elif i % WRITE_THRESHOLD == 0: 
                results = write(results)
                print('Batch written (iteration {})'.format(i)) 

        if results: 
            results = write(results)
        
        output_file.close() 

        print('Got: {} scores'.format((i+1)*SCORE_PER_PAGE))
    
    return file  
            