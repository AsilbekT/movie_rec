import unittest
from movie_recommender.core import MovieRecommender
from movie_recommender.config import MOVIES_CSV, CREDITS_CSV

class TestMovieRecommender(unittest.TestCase):
    def setUp(self):
        self.recommender = MovieRecommender(MOVIES_CSV, CREDITS_CSV)

    def test_recommend(self):
        result = self.recommender.recommend("Avatar")
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
