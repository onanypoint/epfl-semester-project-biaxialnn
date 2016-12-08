MuseScore.com - Python API wrapper
==========

This Python package supports MuseScore.com's REST API (version 1.1) with OAuth 1.0.  It should work with the latest Python versions in both 2.x and 3.x branches.  

Dependencies
---
* Requests http://python-requests.org
* Requests OAuthLib https://github.com/requests/requests-oauthlib

Some Code Examples
------------------
*See test.py for more examples.*

You can use MuseScoreAPI without authentication. For example to list the last scores on MuseScore.com:

    from MuseScoreAPI import MuseScoreAPI
    api = MuseScoreAPI(client_key="musichackday")
    api.request('score')
    print(r.text)

You can also get OAuth token before with connect.py

    python connect.py

It will store the credentials in credentials.json. And you can then use this token to get user details

    from MuseScoreAPI import MuseScoreAPI
    api = MuseScoreAPI("credentials.json")
    api.request('me')
    print(r.text)

Or you can post a score

    from MuseScoreAPI import MuseScoreAPI
    api = MuseScoreAPI("credentials.json")
    r = api.postScore("Title", "test.mscz", description="Description", license=MuseScoreAPI.LICENSE_CC_ZERO)


