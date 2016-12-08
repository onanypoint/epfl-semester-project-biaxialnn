__title__ = 'MuseScpreAPI'
__version__ = '1.0'
__author__ = 'Nicolas Froment'
__license__ = 'MIT'
__copyright__ = 'Copyright 2014 Nicolas Froment'


try:
    #from .MuseScoreOAuth import MuseScoreOAuth
    from .MuseScoreAPI import MuseScoreAPI, MuseScoreResponse, RestIterator
    #from .TwitterRestPager import TwitterRestPager
except:
    pass


__all__ = [
    'MuseScoreAPI',
    #'MuseScoreOAuth'
    #,
    #'MuseScoreRestPager'
]