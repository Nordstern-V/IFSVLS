"""Bounded proposal tests; region geometry alone never grants an identity."""
import unittest
import numpy as np
import pefsl_proposals as proposals


class ProposalTests(unittest.TestCase):
    @unittest.skipIf(proposals.cv2 is None, 'OpenCV unavailable')
    def test_object_bounds_are_proposed_inside_camera_content(self):
        image = np.full((182, 400, 3), 35, np.uint8)
        image[45:125, 140:230] = (40, 70, 220)
        image[58:112:9, 155:215] = (230, 230, 230)
        boxes = proposals.region_proposals(image, limit=6)
        self.assertGreater(len(boxes), 0)
        self.assertLessEqual(len(boxes), 6)
        self.assertTrue(all(0 <= x < r <= 800 and 58 <= y < b <= 422
                            for x, y, r, b in boxes))
        # At least one complete visible object is proposed, rather than only
        # a white stripe from its interior. All still need the B matcher.
        self.assertTrue(any(x <= 284 and y <= 152 and r >= 452 and b >= 302
                            for x, y, r, b in boxes), boxes)

    @unittest.skipIf(proposals.cv2 is None, 'OpenCV unavailable')
    def test_uniform_scene_is_not_an_object_proposal(self):
        self.assertEqual(proposals.region_proposals(np.full((182, 400, 3), 35, np.uint8)), [])

    def test_optional_opencv_falls_back_to_regular_grid(self):
        old = proposals.cv2
        try:
            proposals.cv2 = None
            self.assertEqual(proposals.region_proposals(np.zeros((182, 400, 3), np.uint8)), [])
        finally:
            proposals.cv2 = old


if __name__ == '__main__':
    unittest.main()
