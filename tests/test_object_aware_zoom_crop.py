"""Tests for the object-aware zoom crop used by the Pothole candidate."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from mmdet.structures.bbox import HorizontalBoxes

from research.mmdet_plugins import ObjectAwareZoomCrop, object_aware_crop_window


class ObjectAwareZoomCropTest(unittest.TestCase):

    def test_window_contains_target_and_has_requested_scale(self):
        window = object_aware_crop_window(
            torch.tensor([40.2, 20.1, 80.8, 60.9]),
            (100, 200),
            0.5,
            (0.5, 0.5),
        )
        left, top, right, bottom = window
        self.assertEqual((right - left, bottom - top), (100, 50))
        self.assertLessEqual(left, 40.2)
        self.assertLessEqual(top, 20.1)
        self.assertGreaterEqual(right, 80.8)
        self.assertGreaterEqual(bottom, 60.9)

    def test_probability_zero_is_identity(self):
        transform = ObjectAwareZoomCrop(prob=0.0)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        boxes = HorizontalBoxes(torch.tensor([[40., 20., 80., 60.]]))
        results = transform(dict(img=image.copy(), gt_bboxes=boxes.clone()))
        np.testing.assert_array_equal(results["img"], image)
        torch.testing.assert_close(results["gt_bboxes"].tensor, boxes.tensor)
        self.assertFalse(results["zoom_crop_applied"])

    def test_crop_keeps_target_and_filters_severely_truncated_neighbor(self):
        transform = ObjectAwareZoomCrop(
            prob=1.0,
            crop_scale_range=(0.5, 0.5),
            selection_power=0.5,
            min_retained_area=0.5,
        )
        results = dict(
            img=np.zeros((100, 200, 3), dtype=np.uint8),
            gt_bboxes=HorizontalBoxes(torch.tensor([
                [50., 30., 70., 50.],
                [0., 0., 30., 30.],
            ])),
            gt_bboxes_labels=np.array([3, 7], dtype=np.int64),
            gt_ignore_flags=np.array([False, False]),
        )
        with patch("research.mmdet_plugins.np.random.random", side_effect=[0., .5, .5]), \
                patch("research.mmdet_plugins.np.random.choice", return_value=0), \
                patch("research.mmdet_plugins.np.random.uniform", return_value=.5):
            cropped = transform(results)
        self.assertEqual(cropped["img"].shape[:2], (50, 100))
        self.assertEqual(len(cropped["gt_bboxes"]), 1)
        self.assertEqual(cropped["gt_bboxes_labels"].tolist(), [3])
        self.assertTrue(cropped["zoom_crop_applied"])
        # Relative object area grows fourfold before the common resize.
        original_fraction = 20 * 20 / (100 * 200)
        new_box = cropped["gt_bboxes"].tensor[0]
        new_fraction = float(
            (new_box[2] - new_box[0]) * (new_box[3] - new_box[1]) / (50 * 100))
        self.assertAlmostEqual(new_fraction / original_fraction, 4.0, places=5)

    def test_invalid_settings_and_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            ObjectAwareZoomCrop(prob=1.1)
        with self.assertRaises(ValueError):
            ObjectAwareZoomCrop(crop_scale_range=(0.8, 0.5))
        with self.assertRaises(ValueError):
            ObjectAwareZoomCrop(selection_power=-1)
        with self.assertRaises(ValueError):
            object_aware_crop_window(
                torch.tensor([0., 0., 0., 1.]), (10, 10), .5, (.5, .5))


if __name__ == "__main__":
    unittest.main()
