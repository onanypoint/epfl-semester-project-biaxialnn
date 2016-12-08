from MuseScoreAPI import MuseScoreAPI
import unittest
import sys

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.api = MuseScoreAPI("credentials.json")

    def test_me(self):
        r = self.api.request('me')
        self.assertEqual(r.status_code, 200)
        
        r = self.api.request('me/sets')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('me/scores')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('me/favorites')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('me/favorites')
        self.assertEqual(r.status_code, 200)
    
    def test_me_xml(self):        
        r = self.api.request('me/sets', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<sets'))

        r = self.api.request('me/scores', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<scores'))

        r = self.api.request('me/favorites', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<scores'))

        r = self.api.request('me/activities', format='xml')
        self.assertEqual(r.status_code, 200)
        print r.text
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<scores'))

    def test_user_read(self):
        r = self.api.request('user/:3')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('user/:3/score')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('user/:3/scores')
        self.assertEqual(r.status_code, 200)
        
        r = self.api.request('user/:3/favorites')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('user/:3/followers')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('user/:3/following')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('user/:3/groups')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('user/:3/sets')
        self.assertEqual(r.status_code, 200)

    def test_user_read_xml(self):
        r = self.api.request('user/:3/score', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<scores'))

        r = self.api.request('user/:3/scores', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<scores'))
        
        r = self.api.request('user/:3/favorites', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<scores'))

        r = self.api.request('user/:3/followers', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<users'))

        r = self.api.request('user/:3/following', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<users'))

        r = self.api.request('user/:3/groups', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<groups'))

        r = self.api.request('user/:3/sets', format='xml')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.text.startswith('<?xml version="1.0" encoding="utf-8"?>\n<sets'))

    def test_score_read(self):
        r = self.api.request('score')
        self.assertEqual(r.status_code, 200)
    
        r = self.api.request('score', {"text": "Promenade"})
        self.assertEqual(r.status_code, 200)

        r = self.api.request('score/:46274')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('score/:179821/space')
        self.assertEqual(r.status_code, 200)

        r = self.api.request('score/:179821/time')
        self.assertEqual(r.status_code, 200)

    def test_set(self):
        r = self.api.request('set/:29516')
        self.assertEqual(r.status_code, 200)

#    def test_score_create_delete(self):
#        files = {'score_data': ('test.mscz', open('test.mscz', 'rb'), 'application/octet-stream'),
#            "title": ('',' test'), 
#            "description": ('', 'description'), 
#            "private" : ('', '1')
#        }
#        r = self.api.request('score',  method="POST", files=files)
#        self.assertEqual(r.status_code, 200)
#        score = r.response.json()
#        score_id = score["score_id"]
#
#        r = self.api.request('score/:' + score_id, method="DELETE")
#        self.assertEqual(r.text, "true")
#        self.assertEqual(r.status_code, 200)
#
#    def test_favorite(self):
#        r = self.api.request('score/:46274')
#        self.assertEqual(r.status_code, 200)
#        score = r.response.json()
#        score_id_fav = score["id"]
#        score_fav = score["user_favorite"]
#        self.assertEqual(score_fav, 0)
#
#        r = self.api.request('score/:' + score_id_fav + '/favorite')
#        self.assertEqual(r.text, "true")
#        self.assertEqual(r.status_code, 200)
#
#        r = self.api.request('score/:' + score_id_fav)
#        score = r.response.json()
#        score_id_fav = score["id"]
#        score_fav = score["user_favorite"]
#        self.assertEqual(score_fav, 1)
#
#        r = self.api.request('score/:' + score_id_fav + '/favorite')
#        self.assertEqual(r.text, "true")
#        self.assertEqual(r.status_code, 200)
#
#        r = self.api.request('score/:' + score_id_fav)
#        score = r.response.json()
#        score_id_fav = score["id"]
#        score_fav = score["user_favorite"]
#        self.assertEqual(score_fav, 0)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)