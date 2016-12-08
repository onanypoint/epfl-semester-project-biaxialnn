import codecs
from datetime import datetime
import sys
from MuseScoreAPI import MuseScoreAPI


# These two lines enable debugging at httplib level (requests->urllib3->httplib)
# You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
# The only thing missing will be the response.body which is not logged.
import httplib
import requests
import logging
httplib.HTTPConnection.debuglevel = 1

# You must initialize logging, otherwise you'll not see debug output.
#logging.basicConfig() 
#logging.getLogger().setLevel(logging.DEBUG)
#requests_log = logging.getLogger("requests.packages.urllib3")
#requests_log.setLevel(logging.DEBUG)
#requests_log.propagate = True

try:
    # python 3
    sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)
except:
    # python 2
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)


api = MuseScoreAPI("credentials.json")
#api = MuseScoreAPI(client_key="")


TEST_NUMBER = 2

try:
    if TEST_NUMBER == 0:
        r = api.request('me')
        print(r.text)
        r = api.request('me/favorites')
        print(r.text)
        r = api.request('me/set')
        print(r.text)
        r = api.request('me/groups')
        print(r.text)

    if TEST_NUMBER == 1:
        r = api.request('user/:3')
        print(r.text)

    if TEST_NUMBER == 2:
        r = api.request('user/:5/score', format='xml')
        print(r.text)
        r = api.request('user/:5/favorites', format='xml')
        print(r.text)
        r = api.request('user/:5/set')
        print(r.text)
        r = api.request('user/:5/groups')
        print(r.text)
    
    if TEST_NUMBER == 3:
        r = api.request('score')
        #print(r.text)
        for score in r:
            print score

    if TEST_NUMBER == 4:
        r = api.request('score', {"text": "Promenade"})
        print(r.text)

    if TEST_NUMBER == 5:
        r = api.request('score/:179821')
        print(r.text)

    if TEST_NUMBER == 6:
        r = api.request('score/:179821/space')
        print(r.text)

    if TEST_NUMBER == 7:
        r = api.request('score/:179821/time')
        print(r.text)

    if TEST_NUMBER == 8:
        r = api.request('score/:147837', method="DELETE")
        print(r.text)
    
    if TEST_NUMBER == 9:
        files = {'score_data': ('test.mscz', open('test.mscz', 'rb'), 'application/octet-stream'),
            "title": ('',' test'), 
            "description": ('', 'description'), 
            "private" : ('', '1')
        }
        r = api.request('score',  method="POST", files=files)
        print(r.text)

    if TEST_NUMBER == 10:
        r = api.postScore("title", "test.mscz", description="description", license=MuseScoreAPI.LICENSE_CC_ZERO)
        print(r.text)

except Exception as e:
    print(e)