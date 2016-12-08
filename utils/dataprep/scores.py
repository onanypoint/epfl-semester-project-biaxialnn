import urllib.request
import pandas as pd

def get_filename(id, extension):
    """Get filename based on id
    
    Parameters
    ----------
    id : int
        id of the score
    extension : str
        Extension of the file
    """
    return 'ms_score_' + str(id) + "." + extension

def get_score(directory, id, secret, extension):
    """Retrieve one score from musescore api
    
    Parameters
    ----------
    directory : PosixPath
        Where to save the data (directory).
    id : int
        ID of the score to retrieve
    secret : str
        Secret of the score to retrieve
    extension : str, optional
        Extension of the file
    
    Returns
    -------
    PosixPath
        Score file
    """
    file = directory/get_filename(id, extension) 
    if not file.exists(): 
        url = 'http://static.musescore.com/{}/{}/score.{}'.format(id, secret, extension)
        urllib.request.urlretrieve(url, str(file))
    return file

def get_scores(df, directory, prefix, extension="mxl"):
    """Retrieve scores from musescore api
    
    Note
    ----
    * Uses a Pandas dataframe containing at least the id and secret for each score
    * Will create a directory if given path does not exist.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe containing the information (miniumum: id, secret) about 
        the scores to retrieve
    directory : PosixPath
        Where to save the data (directory).
    extension : str, optional
        Extension of the file (the default is "mxl") mroe extension can be found
        on developer.musescore.com. It is possible that some score are not 
        available in all format.
    
    Returns
    -------
    df : pandas.Dataframe
        Dataframe containing the information (id, secret) about the scores
        to retrieve and the relative file path to the file
    """
    directory = directory/prefix
    directory.mkdir(exist_ok=True)

    print("Retrieving")    
    print("Total", len(df))
    
    def get_score_wrapper(row):
        id = row['id']
        secret = row['secret']

        try:
            file = get_score(directory, id, secret, extension="mxl")
            return pd.Series({'file_name': file.name})
        except :
           print("! Error with score : {}".format(id))
           return pd.Series({'file_name': ''})
    
    df = df.merge(df.apply(lambda row: get_score_wrapper(row), axis=1), left_index=True, right_index=True)
    df.set_index('id', inplace=True)
    
    return df