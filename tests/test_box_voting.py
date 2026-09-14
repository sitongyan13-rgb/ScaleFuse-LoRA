import unittest

import numpy as np

from scripts.search_box_voting import box_vote_predictions


class BoxVotingTest(unittest.TestCase):

    def test_vote_moves_seed_toward_neighbor_without_changing_score(self):
        sources = {
            "a": {1: {"boxes": np.array([[0., 0., 10., 10.]]),
                      "scores": np.array([0.9])}},
            "b": {1: {"boxes": np.array([[1., 1., 11., 11.]]),
                      "scores": np.array([0.8])}},
        }
        result = box_vote_predictions(
            sources, ("a", "b"), 0.5, 0.5, 1.0, top_k=1)[1]
        self.assertEqual(result["scores"].tolist(), [0.9])
        expected = (sources["a"][1]["boxes"][0] * 0.9
                    + sources["b"][1]["boxes"][0] * 0.8) / 1.7
        np.testing.assert_allclose(result["boxes"][0], expected)

    def test_strict_vote_keeps_seed_coordinates(self):
        sources = {
            "a": {1: {"boxes": np.array([[0., 0., 10., 10.]]),
                      "scores": np.array([0.9])}},
            "b": {1: {"boxes": np.array([[5., 5., 15., 15.]]),
                      "scores": np.array([0.8])}},
        }
        result = box_vote_predictions(
            sources, ("a", "b"), 0.5, 0.9, 2.0, top_k=1)[1]
        np.testing.assert_allclose(result["boxes"][0], [0., 0., 10., 10.])

    def test_validation(self):
        with self.assertRaises(ValueError):
            box_vote_predictions({}, (), 0.5, 0.5, 1.0)
        with self.assertRaises(ValueError):
            box_vote_predictions(
                {"a": {1: {"boxes": np.empty((0, 4)),
                            "scores": np.empty(0)}}},
                ("a",), 0.5, 0.5, 0.0)


if __name__ == "__main__":
    unittest.main()
