import unittest

from pycocotools import mask as coco_mask

from just_a_sports_dataset_generator.annotations import (
    AnnotationError,
    ObjectIdentity,
    visible_annotations,
)


class AnnotationTests(unittest.TestCase):
    def test_partial_and_full_occlusion_and_column_major_rle(self):
        pixels = bytes([1, 1, 2, 0, 1, 2, 2, 0, 0, 2, 2, 0])
        objects = (
            ObjectIdentity(1, 0, "ball"),
            ObjectIdentity(2, 1, "player"),
            ObjectIdentity(3, 2, "referee"),
        )
        result = visible_annotations(4, 3, pixels, objects, "basketball")
        self.assertEqual(len(result), 2)  # hidden referee is omitted
        self.assertEqual(result[0].bbox_xywh, (0, 0, 2, 2))
        self.assertEqual(result[0].visible_area, 3)
        self.assertEqual(result[0].rle_counts, (0, 2, 1, 1, 8))
        self.assertEqual(result[1].bbox_xywh, (1, 0, 2, 3))
        self.assertEqual(result[1].visible_area, 5)
        for obj in result:
            encoded = coco_mask.frPyObjects({"size": [3, 4], "counts": list(obj.rle_counts)}, 3, 4)
            decoded = coco_mask.decode(encoded)
            self.assertEqual(
                decoded.flatten().tolist(), [int(p == obj.instance_id) for p in pixels]
            )
            self.assertEqual(int(coco_mask.area(encoded)), obj.visible_area)
            self.assertEqual(tuple(coco_mask.toBbox(encoded)), obj.bbox_xywh)

    def test_background_only_and_full_frame(self):
        self.assertEqual(visible_annotations(1, 1, bytes([0]), (), "basketball"), ())
        obj = visible_annotations(1, 1, bytes([1]), (ObjectIdentity(1, 0, "ball"),), "basketball")[
            0
        ]
        self.assertEqual(obj.bbox_xywh, (0, 0, 1, 1))
        self.assertEqual(obj.rle_counts, (0, 1))

    def test_rejects_invalid_identity_and_buffer(self):
        for pixels, objects in [
            (bytes([2]), (ObjectIdentity(1, 0, "ball"),)),
            (bytes([1]), (ObjectIdentity(1, 0, "player"),)),
            (bytes([1]), (ObjectIdentity(1, 0, "ball"), ObjectIdentity(1, 1, "player"))),
            (bytes([0]), (ObjectIdentity(256, 0, "ball"),)),
            (b"", ()),
        ]:
            with self.subTest(pixels=pixels, objects=objects), self.assertRaises(AnnotationError):
                visible_annotations(1, 1, pixels, objects, "basketball")
