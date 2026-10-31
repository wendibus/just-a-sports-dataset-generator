import unittest

from just_a_sports_dataset_generator.annotations import AnnotationError, ObjectIdentity
from just_a_sports_dataset_generator.annotations.poses import OPENPOSE_18, frame_poses

WIDTH, HEIGHT = 8, 6
OBJECTS = (
    ObjectIdentity(1, 0, "ball"),
    ObjectIdentity(10, 1, "player"),
    ObjectIdentity(30, 2, "referee"),
)


def ids() -> bytes:
    pixels = bytearray(WIDTH * HEIGHT)
    for y in range(HEIGHT):
        for x in range(4):
            pixels[y * WIDTH + x] = 10  # player covers the left half
    pixels[1 * WIDTH + 6] = 30  # one referee pixel
    return bytes(pixels)


def person(identity=10, **overrides):
    keypoints = [[1.0, 1.0]] * len(OPENPOSE_18)
    for index, value in overrides.items():
        keypoints[int(index.removeprefix("k"))] = value
    return {"instance_id": identity, "keypoints": keypoints}


class PoseVisibilityTests(unittest.TestCase):
    def test_coco_visibility_from_id_pass_and_facing(self):
        raw = [
            person(
                k1=[6.0, 4.0],  # inside the image but not on the player: occluded
                k2=[-3.0, 2.0],  # outside the image
                k3=None,  # behind the camera
                k4=[1.0, 1.0, False],  # turned away from the camera
                k5=[1.0, 1.0, True],
                k6=[4.4, 2.0],  # rounding at a thin limb: neighbourhood reaches the player
            )
        ]
        [result] = frame_poses(raw, OBJECTS, WIDTH, HEIGHT, ids())
        flags = [point[2] for point in result["keypoints"]]  # type: ignore[index]
        self.assertEqual(result["label"], "player")
        self.assertEqual(flags[:7], [2, 1, 0, 0, 1, 2, 2])
        self.assertEqual(result["keypoints"][2], [0.0, 0.0, 0])  # type: ignore[index]

    def test_invalid_worker_poses_are_rejected(self):
        cases = {
            "ball has no pose": [person(identity=1)],
            "unknown instance": [person(identity=99)],
            "repeated person": [person(), person()],
            "wrong count": [{"instance_id": 10, "keypoints": [[1.0, 1.0]]}],
            "non-finite": [person(k0=[float("nan"), 1.0])],
            "bad facing flag": [person(k0=[1.0, 1.0, 1])],
            "bad shape": [person(k0=[1.0])],
        }
        for name, raw in cases.items():
            with self.subTest(name), self.assertRaises(AnnotationError):
                frame_poses(raw, OBJECTS, WIDTH, HEIGHT, ids())
        with self.assertRaises(AnnotationError):
            frame_poses([], OBJECTS, WIDTH, HEIGHT, b"\0")


if __name__ == "__main__":
    unittest.main()
