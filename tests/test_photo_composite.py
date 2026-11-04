import unittest

from PIL import Image

from just_a_sports_dataset_generator.annotations import ObjectIdentity, visible_annotations
from just_a_sports_dataset_generator.photo_composite import PhotoLayer, compose_photo_layers


class PhotoCompositeTests(unittest.TestCase):
    def test_final_visible_pixels_follow_depth_and_alpha_holes(self):
        background = Image.new("RGB", (40, 40), "white")
        ids = Image.new("L", background.size)
        ids.paste(1, (0, 0, 4, 4))
        base = (ObjectIdentity(1, 0, "ball"),)
        red = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        blue = Image.new("RGBA", (16, 16), (0, 0, 255, 255))
        blue.putpixel((8, 8), (0, 0, 255, 0))
        layers = [
            PhotoLayer(ObjectIdentity(11, 1, "player"), blue, (20, 30), 16),
            PhotoLayer(ObjectIdentity(10, 1, "player"), red, (20, 25), 16),
        ]
        rgb, final = compose_photo_layers(background, ids, base, layers, shadows=False)
        self.assertEqual(final.getpixel((20, 22)), 10)  # Hole in front player reveals rear.
        self.assertEqual(rgb.getpixel((20, 22)), (255, 0, 0))
        self.assertEqual(final.getpixel((19, 22)), 11)
        self.assertEqual(rgb.getpixel((19, 22)), (0, 0, 255))
        self.assertEqual(final.getpixel((0, 0)), 1)
        raw = rgb.tobytes()
        triples = zip(raw[::3], raw[1::3], raw[2::3])
        for pixel, identity in zip(triples, final.tobytes()):
            if identity == 10:
                self.assertEqual(pixel, (255, 0, 0))
            elif identity == 11:
                self.assertEqual(pixel, (0, 0, 255))
        annotations = visible_annotations(
            40, 40, final.tobytes(), base + tuple(x.identity for x in layers), "basketball"
        )
        self.assertEqual(
            sum(a.visible_area for a in annotations), sum(v != 0 for v in final.tobytes())
        )
        self.assertEqual(ids.getpixel((20, 22)), 0)  # Inputs remain unchanged.

    def test_clipping_and_complete_occlusion_remove_hidden_annotation(self):
        background = Image.new("RGB", (20, 20))
        ids = Image.new("L", background.size)
        ids.paste(1, (0, 0, 3, 3))
        person = PhotoLayer(
            ObjectIdentity(10, 1, "player"), Image.new("RGBA", (16, 16), "red"), (2, 10), 16
        )
        rgb, final = compose_photo_layers(
            background, ids, (ObjectIdentity(1, 0, "ball"),), [person], shadows=False
        )
        annotations = visible_annotations(
            *rgb.size,
            final.tobytes(),
            (person.identity, ObjectIdentity(1, 0, "ball")),
            "basketball",
        )
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].bbox_xywh, (0, 0, 10, 10))
        self.assertEqual(annotations[0].visible_area, 100)

    def test_rejects_missing_alpha_empty_mask_duplicate_ids_and_ambiguous_depth(self):
        background = Image.new("RGB", (20, 20))
        ids = Image.new("L", background.size)
        identity = ObjectIdentity(10, 1, "player")
        for image in [Image.new("RGB", (16, 16)), Image.new("RGBA", (16, 16))]:
            with self.assertRaises(ValueError):
                compose_photo_layers(
                    background, ids, (), [PhotoLayer(identity, image, (10, 20), 16)]
                )
        layer = PhotoLayer(identity, Image.new("RGBA", (16, 16), "red"), (10, 20), 16)
        with self.assertRaises(ValueError):
            compose_photo_layers(background, ids, (), [layer, layer])
        other = PhotoLayer(ObjectIdentity(11, 1, "player"), layer.rgba, layer.foot, 16)
        with self.assertRaises(ValueError):
            compose_photo_layers(background, ids, (), [layer, other])
