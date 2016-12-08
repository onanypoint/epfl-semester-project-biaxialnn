"""
    Constants For All MuseScore Endpoints
    -----------------------------------
    
    Version 1.0,EST API.
    
    URLs for each endpoint are composed of the following pieces:
        PROTOCOL://{subdomain}.DOMAIN/VERSION/{resource}?{parameters}
"""

__author__ = "Nicolas Froment"
__date__ = "April 12, 2014"
__license__ = "MIT"


PROTOCOL = 'http'

DOMAIN = 'musescore.com'

VERSION = 'services/rest'

USER_AGENT = 'python-MuseScoreAPI'

STREAMING_SOCKET_TIMEOUT = 90  # 90 seconds per Twitter's recommendation


REST_SUBDOMAIN = 'api'

REST_SOCKET_TIMEOUT = 5

REST_ENDPOINTS = {
        # resource:              ( method )
        'me':                    ('GET',),
        'me/sets':               ('GET',),
        'me/scores':             ('GET',),
        'me/favorites':          ('GET',),
        'me/activities':         ('GET',),
        'me/history':            ('GET',),
        
        'user/:PARAM':           ('GET', 'POST'),  # ID 
        'user/:PARAM/score':     ('GET',),         # ID
        'user/:PARAM/scores':    ('GET',),         # ID
        'user/:PARAM/favorites': ('GET',),         # ID 
        'user/:PARAM/followers': ('GET',),         # ID 
        'user/:PARAM/following': ('GET',),         # ID 
        'user/:PARAM/groups':    ('GET',),         # ID
        'user/:PARAM/follow':    ('GET',),         # ID
        'user/:PARAM/sets':      ('GET',),         # ID


        'score':                 ('GET', 'POST'),  
        'score/:PARAM':          ('GET', 'DELETE'), # ID  
        'score/:PARAM/time':     ('GET',),   # ID 
        'score/:PARAM/space':    ('GET',),
        'score/:PARAM/favorite': ('GET',),   # ID           
        'score/:PARAM/comments': ('GET',),   # ID
        'score/:PARAM/comment':  ('POST',),   # ID
        'score/:PARAM/update':   ('POST',),   # ID

        'set/:PARAM':             ('GET', 'DELETE'),     # ID 

        'groups':                 ('GET',),  
        'groups/:PARAM':          ('GET', 'DELETE'),    # ID  
        'groups/:PARAM/score':    ('GET',),              # ID 

        'resolve':                ('GET',)
}