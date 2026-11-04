import unittest

from PIL import Image, ImageDraw

from just_a_sports_dataset_generator.foreground_quality import coverage_metrics


class ForegroundQualityTests(unittest.TestCase):
    def masks(self):
        mask = Image.new("L", (200, 300))
        draw = ImageDraw.Draw(mask)
        draw.rectangle((70, 30, 130, 200), fill=255)
        draw.rectangle((10, 80, 70, 95), fill=255)
        draw.rectangle((70, 200, 90, 280), fill=255)
        draw.rectangle((110, 200, 130, 280), fill=255)
        return mask

    def test_exact_foreground_and_small_edge_shift(self):
        ref = self.masks()
        self.assertEqual(coverage_metrics(ref, ref)["min_local_coverage"], 1)
        shifted = ref.transform(ref.size, Image.Transform.AFFINE, (1, 0, 1, 0, 1, 0))
        self.assertGreater(coverage_metrics(ref, shifted)["foreground_coverage"], 0.99)

    def test_missing_hand_is_not_hidden_by_body_area(self):
        ref = self.masks()
        fg = ref.copy()
        ImageDraw.Draw(fg).rectangle((0, 65, 55, 110), fill=0)
        metrics = coverage_metrics(ref, fg)
        self.assertGreater(metrics["foreground_coverage"], 0.9)
        self.assertLess(metrics["min_local_coverage"], 0.55)

    def test_missing_leg_and_empty_masks(self):
        ref = self.masks()
        fg = ref.copy()
        ImageDraw.Draw(fg).rectangle((100, 210, 140, 300), fill=0)
        self.assertLess(coverage_metrics(ref, fg)["min_local_coverage"], 0.55)
        self.assertEqual(coverage_metrics(ref, Image.new("L", ref.size))["foreground_coverage"], 0)
        self.assertEqual(coverage_metrics(Image.new("L", ref.size), fg)["foreground_coverage"], 0)
        with self.assertRaises(ValueError):
            coverage_metrics(ref, Image.new("L", (1, 1)))
