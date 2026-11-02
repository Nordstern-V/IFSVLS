"""Host tests for bounded C124-inspired spatial identity checks."""

import unittest

import numpy as np

from pefsl_identity import (
    identity_evidence, prepare_reference, shape_descriptor,
)


def patterned(seed=4):
    rng = np.random.RandomState(seed)
    # Textured but spatially coherent shapes cover all nine support cells.
    cells = rng.randint(25, 190, (12, 12)).astype(np.uint8)
    gray = np.repeat(np.repeat(cells, 4, axis=0), 4, axis=1)
    return np.repeat(gray[:, :, None], 3, axis=2)


class IdentityTests(unittest.TestCase):
    def test_exact_and_exposure_shifted_registration_match(self):
        ref = patterned()
        self.assertTrue(identity_evidence(ref, [ref])["supported"])
        shifted = np.clip(ref.astype(np.float32) * .8 + 25., 0, 255).astype(np.uint8)
        evidence = identity_evidence(shifted, [ref])
        self.assertTrue(evidence["supported"], evidence)
        self.assertGreaterEqual(evidence["cells"], 4)

    def test_quarter_turn_and_small_translation_match(self):
        ref = patterned()
        rotated = np.ascontiguousarray(np.rot90(ref))
        evidence = identity_evidence(rotated, [ref])
        self.assertTrue(evidence["supported"], evidence)
        moved = np.pad(ref, ((0, 2), (0, 2), (0, 0)), mode="edge")[2:, 2:]
        self.assertTrue(identity_evidence(moved, [ref])["supported"])

    def test_same_colour_distribution_different_layout_rejected(self):
        ref = patterned()
        # The same pixels/colour histogram, shuffled independently by row.
        different = ref[np.random.RandomState(12).permutation(48)]
        evidence = identity_evidence(different, [ref])
        self.assertFalse(evidence["supported"], evidence)

    def test_small_shared_patch_does_not_prove_whole_roi(self):
        ref = patterned()
        other = patterned(seed=15)
        other[16:32, 16:32] = ref[16:32, 16:32]
        self.assertFalse(identity_evidence(other, [ref])["supported"])

    def test_flat_reference_cannot_supply_spatial_identity(self):
        flat = np.full((48, 48, 3), 170, dtype=np.uint8)
        evidence = identity_evidence(flat, [flat])
        self.assertFalse(evidence["supported"])
        self.assertEqual(evidence["reason"], "textureless reference")
        self.assertFalse(evidence["textured"])

    def test_legacy_no_references_are_explicitly_undecided(self):
        evidence = identity_evidence(patterned(), [])
        self.assertIsNone(evidence["supported"])

    def test_reference_format_and_shape_descriptor(self):
        image = np.tile(patterned(), (2, 3, 1))
        prepared = prepare_reference(image)
        self.assertEqual(prepared.shape, (48, 48, 3))
        self.assertEqual(prepared.dtype, np.uint8)
        shape = shape_descriptor(patterned())
        self.assertEqual(shape.shape, (2,))
        self.assertTrue(np.isfinite(shape).all())
        self.assertGreater(shape[0], .01)
        self.assertGreater(shape[1], 8.)
        np.testing.assert_array_equal(
            shape_descriptor(np.full((48, 48, 3), 170, dtype=np.uint8)),
            np.zeros(2, dtype=np.float32),
        )

    def test_reference_validation(self):
        with self.assertRaises(ValueError):
            prepare_reference(np.empty((0, 0, 3)))
        with self.assertRaises(ValueError):
            prepare_reference(np.zeros((8, 8)))


if __name__ == "__main__":
    unittest.main()
