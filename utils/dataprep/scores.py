import urllib.request
import pandas as pd

def get_filename(id, extension):
    return 'ms_score_' + str(id) + "." + extension

def get_score(directory, id, secret, extension):
    file = directory/get_filename(id, extension) 
    if not file.exists(): 
        url = 'http://static.musescore.com/{}/{}/score.{}'.format(id, secret, extension)
        urllib.request.urlretrieve(url, str(file))
    return file

def get_scores(df, directory, extension="mxl"):
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