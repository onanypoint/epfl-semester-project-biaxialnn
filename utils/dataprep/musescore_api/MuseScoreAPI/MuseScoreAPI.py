__author__ = "Jonas Geduldig"
__date__ = "June 7, 2013"
__license__ = "MIT"

from .constants import *
import json
from requests_oauthlib import OAuth1
from datetime import datetime
import requests
import json
import os.path


class MuseScoreAPI(object):

    LICENSE_ALL_RIGHT_RESERVED = "all-rights-reserved"
    LICENSE_CC_BY = "cc-by"
    LICENSE_CC_BY_SA = "cc-by-sa" 
    LICENSE_CC_BY_ND = "cc-by-nd"
    LICENSE_CC_BY_NC = "cc-by-nc"
    LICENSE_CC_BY_NC_SA = "cc-by-nc-sa"
    LICENSE_CC_BY_NC_ND = "cc-by-nc-nd"
    LICENSE_PD = "publicdomain"
    LICENSE_CC_ZERO = "cc-zero"

    """Access REST API or Streaming API resources.

    :param consumer_key: MuseScore application consumer key
    :param consumer_secret: MuseScore application consumer secret
    :param access_token_key: MuseScore application access token key
    :param access_token_secret: MuseScore application access token secret
    :param proxy_url: HTTPS proxy URL (ex. "https://USER:PASSWORD@SERVER:PORT")
    """

    def __init__(
            self,
            credFile=None,
            client_key=None,
            proxy_url=None):
        """Initialize with your MuseScore application credentials"""
        self.proxies = {'https': proxy_url} if proxy_url else None
        auth_type='oAuth1'
        if credFile and os.path.isfile(credFile):
            with open("credentials.json") as json_file:
                cred = json.load(json_file)
            self.auth = OAuth1(cred["client_key"],
                          client_secret=cred["client_secret"],
                          resource_owner_key=cred["resource_owner_key"],
                          resource_owner_secret=cred["resource_owner_secret"])
        elif client_key:
            self.auth = None
            self.client_key = client_key
        else:
            raise Exception('At least a client key is needed')


    def _prepare_url(self, subdomain, path, format):
        return '%s://%s.%s/%s/%s.%s' % (PROTOCOL,
                                          subdomain,
                                          DOMAIN,
                                          VERSION,
                                          path,
                                          format)

    def _get_endpoint(self, resource):
        """Substitute any parameters in the resource path with :PARAM."""
        if ':' in resource:
            parts = resource.split('/')
            # embedded parameters start with ':'
            parts = [k if k[0] != ':' else ':PARAM' for k in parts]
            endpoint = '/'.join(parts)
            resource = resource.replace(':', '')
            return (resource, endpoint)
        else:
            return (resource, resource)

    def request(self, resource, params=None, method='GET', files=None, format='json'):
        """Request a MuseScore REST API resource.

        :param resource: A valid MuseScore endpoint (ex. "score")
        :param params: Dictionary with endpoint parameters or None (default)
        :param method: String the method to use. GET (default)
        :param files: Dictionary with multipart-encoded file or None (default)

        :returns: MuseScoreAPI.MuseScoreResponse object
        """
        session = requests.Session()
        if self.auth:
            session.auth = self.auth
        elif self.client_key:
            if not params:
                params = {}
            params['oauth_consumer_key'] = self.client_key
        session.headers = {'User-Agent': USER_AGENT}
        resource, endpoint = self._get_endpoint(resource)
        if endpoint in REST_ENDPOINTS:
            session.stream = False
            methods = REST_ENDPOINTS[endpoint]
            if not method in (name.upper() for name in methods):
                raise Exception('"%s" is not valid endpoint for resource "%s"' % (method, resource))
            url = self._prepare_url(REST_SUBDOMAIN, resource, format)
            timeout = REST_SOCKET_TIMEOUT
        else:
            raise Exception('"%s" is not a valid endpoint' % resource)
        r = session.request(
            method,
            url,
            params=params,
            timeout=timeout,
            files=files,
            proxies=self.proxies)
        return MuseScoreResponse(r, session.stream)

    def postScore(self, title, file, private=1, description="", tags="", license=LICENSE_ALL_RIGHT_RESERVED):
        """Post a score on MuseScore.com REST

        :param title: the title of the piece
        :param file: File path to a mscz file
        :param private: 0 or 1 depending if the score is public or private
        :param description: Description of the score
        :param tags: Comma separated list of tags
        :param license: one of the license from LICENSE_ALL_RIGHT_RESERVED, LICENSE_CC_BY,
        LICENSE_CC_BY_SA, LICENSE_CC_BY_ND, LICENSE_CC_BY_NC, LICENSE_CC_BY_NC_SA,
        LICENSE_CC_BY_NC_ND, LICENSE_PD or LICENSE_CC_ZERO

        :returns: MuseScoreAPI.MuseScoreResponse object
        """
        if not os.path.isfile(file):
            raise Exception('"%s" not found' % file)
        if not os.path.splitext(file)[1].lower() == ".mscz":
            raise Exception('"%s" is not an MSCZ file' % file)
        filename = os.path.basename(file)
        files = {'score_data': (filename, open(file, 'rb'), 'application/octet-stream'),
            "title": ('', title), 
            "description": ('', description), 
            "private" : ('', str(private)),
            "tags" : ('', tags),
            "license" : ('', license),
        }
        return self.request('score',  method="POST", files=files)


class MuseScoreResponse(object):

    """Response from a REST API resource call.

    :param response: The requests.Response object returned by the API call
    """

    def __init__(self, response, stream):
        self.response = response

    @property
    def headers(self):
        """:returns: Dictionary of API response header contents."""
        return self.response.headers

    @property
    def status_code(self):
        """:returns: HTTP response status code."""
        return self.response.status_code

    @property
    def text(self):
        """:returns: Raw API response text."""
        return self.response.text

    def get_iterator(self):
        """:returns:  MuseScoreAPI.RestIterator."""
        return RestIterator(self.response)

    def __iter__(self):
        for item in self.get_iterator():
            yield item

class RestIterator(object):

    """Iterate statuses, errors or other iterable objects in a REST API response.

    :param response: The request.Response from a MuseScore REST API request
    """

    def __init__(self, response):
        resp = response.json()
        if hasattr(resp, '__iter__') and not isinstance(resp, dict):
            self.results = resp
        else:
            self.results = (resp,)

    def __iter__(self):
        """Return a score as a JSON object."""
        for item in self.results:
            yield item
            