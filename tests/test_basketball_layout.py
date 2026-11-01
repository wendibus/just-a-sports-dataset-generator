import json
import math
import random
import unittest

from just_a_sports_dataset_generator.sports.basketball import (
    ANKLE_Z,
    BENCH_Y,
    FOREARM,
    OPENPOSE_18,
    SHIN,
    SITUATIONS,
    THIGH,
    UPPER_ARM,
    Root,
    plan_scene,
    sample_pose,
    skeleton,
)

FAMILIES = ("stand", "run", "defend", "dribble", "shoot", "pass", "catch", "rebound", "signal")


def distance(a, b):
    return math.dist(a, b)


class LayoutTests(unittest.TestCase):
    def test_plans_are_deterministic_and_json_safe(self):
        first, again, other = plan_scene(7, 4), plan_scene(7, 4), plan_scene(8, 4)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        json.dumps(first, allow_nan=False)
        self.assertEqual(len(first["ball"]["positions"]), 4)  # type: ignore[index]

    def test_every_situation_and_ball_event_occurs(self):
        plans = [plan_scene(seed, 3) for seed in range(300)]
        self.assertEqual({p["situation"] for p in plans}, set(SITUATIONS))
        events = {p["ball"]["event"] for p in plans}  # type: ignore[index]
        self.assertEqual(events, {"held", "pass", "shot", "rim", "floor", "toss"})
        self.assertEqual({p["attack_direction"] for p in plans}, {-1, 1})
        # Both halves and the middle of the court are used, not one fixed formation.
        xs = [a["frames"][0]["position"][0] for p in plans for a in p["athletes"]]  # type: ignore[index]
        self.assertLess(min(xs), -10)
        self.assertGreater(max(xs), 10)

    def test_skeletons_keep_bone_lengths_and_feet_on_the_floor(self):
        rng = random.Random(1)
        for family in FAMILIES:
            for _ in range(40):
                joints = skeleton(sample_pose(rng, family))
                self.assertEqual(set(OPENPOSE_18) - set(joints), set())
                for side in ("right", "left"):
                    with self.subTest(family=family, side=side):
                        self.assertAlmostEqual(
                            distance(joints[f"{side}_shoulder"], joints[f"{side}_elbow"]),
                            UPPER_ARM,
                            6,
                        )
                        self.assertAlmostEqual(
                            distance(joints[f"{side}_elbow"], joints[f"{side}_wrist"]), FOREARM, 6
                        )
                        self.assertAlmostEqual(
                            distance(joints[f"{side}_hip"], joints[f"{side}_knee"]), THIGH, 4
                        )
                        self.assertAlmostEqual(
                            distance(joints[f"{side}_knee"], joints[f"{side}_ankle"]), SHIN, 4
                        )
                        self.assertAlmostEqual(joints[f"{side}_ankle"][2], ANKLE_Z, delta=0.02)
                        # Knees bend forward (athletes face -y), never backwards.
                        hip, knee, ankle = (joints[f"{side}_{j}"] for j in ("hip", "knee", "ankle"))
                        self.assertLessEqual(knee[1], (hip[1] + ankle[1]) / 2 + 1e-6)

    def test_athletes_are_spread_out_and_labelled(self):
        for seed in range(150):
            plan = plan_scene(seed, 2)
            athletes = plan["athletes"]
            ids = [a["instance_id"] for a in athletes]  # type: ignore[index]
            self.assertEqual(ids[:10], list(range(10, 20)))
            self.assertTrue(set(ids[10:]) <= {30, 31} and ids[10:])
            labels = {a["instance_id"]: a["label"] for a in athletes}  # type: ignore[index]
            self.assertEqual({labels[i] for i in range(10, 20)}, {"player"})
            spots = [a["frames"][0]["position"][:2] for a in athletes[:10]]  # type: ignore[index]
            closest = min(distance(a, b) for i, a in enumerate(spots) for b in spots[i + 1 :])
            with self.subTest(seed=seed, situation=plan["situation"]):
                self.assertGreaterEqual(closest, 0.6)
                for x, y in spots:
                    self.assertLessEqual(abs(x), 15.0)
                    self.assertLessEqual(abs(y), 8.6)

    def test_a_held_ball_sits_in_the_holders_hands(self):
        checked = 0
        for seed in range(200):
            plan = plan_scene(seed, 3)
            ball = plan["ball"]
            if ball["event"] != "held":  # type: ignore[index]
                continue
            holder = next(a for a in plan["athletes"] if a["instance_id"] == ball["holder"])  # type: ignore[index]
            for frame, position in enumerate(ball["positions"]):  # type: ignore[index]
                f = holder["frames"][frame]
                root = Root(*f["position"], f["heading"], f["height_scale"])
                wrists = [root.world(tuple(f["joints"][f"{s}_wrist"])) for s in ("right", "left")]
                # Horizontally at a hand; vertically anywhere between the floor (dribble
                # bounce) and the hands.
                if holder["pose"]["family"] == "dribble":
                    self.assertLess(min(distance(position[:2], w[:2]) for w in wrists), 0.1)
                else:  # held with both hands, between and slightly in front of them
                    middle = [(a + b) / 2 for a, b in zip(*wrists, strict=True)]
                    self.assertLess(distance(position[:2], middle[:2]), 0.15)
                self.assertLessEqual(position[2], max(w[2] for w in wrists) + 0.05)
                self.assertGreaterEqual(position[2], 0.119)  # never below the floor
            checked += 1
        self.assertGreater(checked, 20)

    def test_motion_between_frames_and_static_free_throws(self):
        moving = [
            p for p in (plan_scene(s, 3) for s in range(60)) if p["situation"] == "transition"
        ]
        self.assertTrue(moving)
        plan = moving[0]
        start = [a["frames"][0]["position"] for a in plan["athletes"]]  # type: ignore[index]
        later = [a["frames"][2]["position"] for a in plan["athletes"]]  # type: ignore[index]
        self.assertGreater(max(distance(a, b) for a, b in zip(start, later, strict=True)), 0.5)
        for seed in range(80):
            plan = plan_scene(seed, 3)
            if plan["situation"] == "free_throw":
                shooter = plan["athletes"][0]  # type: ignore[index]
                self.assertEqual(shooter["frames"][0]["position"], shooter["frames"][2]["position"])

    def test_animated_frames_keep_bone_lengths_and_floor_contact(self):
        bones = (
            ("shoulder", "elbow", UPPER_ARM),
            ("elbow", "wrist", FOREARM),
            ("hip", "knee", THIGH),
            ("knee", "ankle", SHIN),
        )
        for seed in range(60):
            for athlete in plan_scene(seed, 4)["athletes"]:  # type: ignore[union-attr]
                for frame in athlete["frames"]:
                    joints = frame["joints"]
                    for side in ("right", "left"):
                        for a, b, length in bones:
                            self.assertAlmostEqual(
                                distance(joints[f"{side}_{a}"], joints[f"{side}_{b}"]), length, 3
                            )
                        self.assertAlmostEqual(joints[f"{side}_ankle"][2], ANKLE_Z, delta=0.02)
                    self.assertGreaterEqual(frame["position"][2], 0.0)

    def test_runners_cycle_their_legs_and_face_where_they_run(self):
        checked = 0
        for seed in range(80):
            plan = plan_scene(seed, 4)
            for athlete in plan["athletes"]:  # type: ignore[union-attr]
                frames = athlete["frames"]
                start, end = frames[0]["position"], frames[-1]["position"]
                if athlete["pose"]["family"] != "run" or distance(start, end) < 1.5:
                    continue
                # The right ankle moves forward and back relative to the body.
                strides = [f["joints"]["right_ankle"][1] for f in frames]
                self.assertGreater(max(strides) - min(strides), 0.1)
                dx, dy = end[0] - start[0], end[1] - start[1]
                heading = frames[0]["heading"]
                forward = (math.sin(heading), -math.cos(heading))  # local -y in world
                cosine = (forward[0] * dx + forward[1] * dy) / math.hypot(dx, dy)
                self.assertGreater(cosine, 0.9)
                checked += 1
        self.assertGreater(checked, 10)

    def test_jumps_follow_a_ballistic_arc(self):
        heights = []
        for seed in range(120):
            for athlete in plan_scene(seed, 6)["athletes"]:  # type: ignore[union-attr]
                zs = [f["position"][2] for f in athlete["frames"]]
                if max(zs) > 0:
                    heights.append(zs)
                    self.assertLessEqual(max(zs), 0.46)
        self.assertGreater(len(heights), 10)
        # Athletes are not frozen in the air: most jumps change height between frames.
        self.assertGreater(sum(len(set(zs)) > 1 for zs in heights) / len(heights), 0.8)

    def test_a_shot_leaves_the_hands_at_the_release(self):
        checked = 0
        for seed in range(400):
            plan = plan_scene(seed, 6)
            ball = plan["ball"]
            if (
                ball["event"] != "shot"
                or "held" not in ball["states"]
                or "air" not in ball["states"]
            ):  # type: ignore[index]
                continue
            states = ball["states"]  # type: ignore[index]
            first_air = states.index("air")
            self.assertTrue(all(state == "held" for state in states[:first_air]))
            # Once released, the ball rises above the shooter's set point.
            self.assertGreater(ball["positions"][first_air][2], 1.8)  # type: ignore[index]
            checked += 1
        self.assertGreater(checked, 3)

    def test_benches_are_beside_the_court_and_unlabelled(self):
        found = 0
        for seed in range(60):
            plan = plan_scene(seed, 2)
            athletes = plan["athletes"]  # type: ignore[assignment]
            for person in plan["benches"]:  # type: ignore[union-attr]
                found += 1
                x, y = person["position"]
                self.assertGreater(y, 8.2)
                self.assertLessEqual(y, BENCH_Y)
                self.assertNotIn("instance_id", person)
                for athlete in athletes:  # type: ignore[union-attr]
                    for frame in athlete["frames"]:
                        ax, ay, _ = frame["position"]
                        self.assertGreater(math.hypot(ax - x, ay - y), 0.4)
        self.assertGreater(found, 50)


if __name__ == "__main__":
    unittest.main()
