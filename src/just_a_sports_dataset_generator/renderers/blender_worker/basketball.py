# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender-only procedural basketball scene. No external assets or downloads.

Invoked by the Apache-2.0 host through a JSON/file boundary. Object Index is
exported from the SAME Cycles render as RGB, without color management or AA.
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def material(name, color, roughness=0.5, metallic=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat


def finish(obj, name, mat, identity=0, parent=None):
    obj.name = name
    obj.data.materials.append(mat)
    obj.pass_index = identity
    if parent is not None:
        obj.parent = parent
    return obj


def box(name, location, size, mat, identity=0, parent=None, bevel=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = finish(bpy.context.object, name, mat, identity, parent)
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new("Soft manufactured edges", "BEVEL")
        mod.width, mod.segments = bevel, 3
        obj.modifiers.new("Weighted surface normals", "WEIGHTED_NORMAL")
    return obj


def oval(name, location, scale, mat, identity=0, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=location)
    obj = finish(bpy.context.object, name, mat, identity, parent)
    obj.scale = scale
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def limb(name, a, b, r1, r2, mat, identity=0, parent=None):
    a, b = Vector(a), Vector(b)
    delta = b - a
    bpy.ops.mesh.primitive_cone_add(
        vertices=20, radius1=r1, radius2=r2, depth=delta.length, location=(a + b) * 0.5
    )
    obj = finish(bpy.context.object, name, mat, identity, parent)
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def line(name, points, mat, radius=0.025, identity=0, parent=None, closed=False):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.bevel_depth, data.bevel_resolution = radius, 2
    spline = data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, value in zip(spline.points, points):
        point.co = (*value, 1)
    spline.use_cyclic_u = closed
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return finish(obj, name, mat, identity, parent)


def circle(name, center, radius, mat, width=0.025, identity=0, parent=None):
    return line(
        name,
        [
            (
                center[0] + radius * math.cos(i * math.tau / 96),
                center[1] + radius * math.sin(i * math.tau / 96),
                center[2],
            )
            for i in range(96)
        ],
        mat,
        width,
        identity,
        parent,
        True,
    )


def text_object(body, location, scale, mat, rotation=(math.pi / 2, 0, 0), identity=0, parent=None):
    data = bpy.data.curves.new("Lettering", "FONT")
    data.body, data.align_x, data.size = body, "CENTER", scale
    data.extrude = 0.0008
    obj = bpy.data.objects.new("Lettering " + body, data)
    bpy.context.collection.objects.link(obj)
    obj.location, obj.rotation_euler = location, rotation
    return finish(obj, obj.name, mat, identity, parent)


def wood_material(color, gloss):
    mat = material("Maple varnish, procedural grain", color, 0.29)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Coat Weight"].default_value = gloss
    coord = nodes.new("ShaderNodeTexCoord")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = (2, 130, 4)
    links.new(coord.outputs["Generated"], scale.inputs[0])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4
    noise.inputs["Detail"].default_value = 3
    links.new(scale.outputs["Vector"], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*(c * 0.42 for c in color), 1)
    ramp.color_ramp.elements[1].color = (*(min(c * 1.2, 1) for c in color), 1)
    links.new(noise.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs[0], shader.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.008
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs[0], shader.inputs["Normal"])
    return mat


SEAT_ROWS, SEAT_COLUMNS = 4, 42


def seat_position(row, col):
    return -16.4 + col * 0.8, 10.4 + row * 0.82, 0.30 + row * 0.43


def arena(hall):
    """Court, stands, walls and lights; all colours and light settings come from the layout."""
    white = material("Court lines", tuple(hall["line_color"]), 0.38)
    key = material("Painted key", tuple(hall["key_color"]), 0.38)
    concrete = material("Wall", tuple(hall["wall_color"]), 0.8)
    steel = material("Powder coated steel", (0.08, 0.10, 0.13), 0.5, 0.35)
    wood = hall.get("floor_kind", "wood") == "wood"
    if wood:
        floor = wood_material(tuple(hall["floor_color"]), hall["floor_gloss"])
    else:  # poured polyurethane sports floor: even colour, slight sheen, no boards
        floor = material("Synthetic sports floor", tuple(hall["floor_color"]), 0.55)
        floor.node_tree.nodes.get("Principled BSDF").inputs["Coat Weight"].default_value = hall[
            "floor_gloss"
        ]
    box("Court floor", (0, 0, -0.11), (44, 30, 0.2), floor)
    if wood:  # fine plank seams, all coplanar and outside the annotation classes
        seam = material("Fine wood joints", tuple(c * 0.35 for c in hall["floor_color"]), 0.8)
        for row in range(55):
            y = -10.8 + row * 0.4
            line("Maple board joint", [(-17, y, -0.008), (17, y, -0.008)], seam, 0.003)
            for col in range(9):
                x = -16 + col * 4 + (row % 3) * 1.2
                line("Staggered board end", [(x, y, -0.007), (x, y + 0.4, -0.007)], seam, 0.002)
    for extra in hall.get("extra_lines", []):
        other_sport_lines(
            extra["sport"], material(f"{extra['sport']} lines", tuple(extra["color"]), 0.4)
        )
    for sign in (-1, 1):
        box("Painted key", (sign * 11.1, 0, 0.001), (5.8, 4.9, 0.004), key)
        line(
            "Key boundary",
            [
                (sign * 14, -2.45, 0.012),
                (sign * 8.2, -2.45, 0.012),
                (sign * 8.2, 2.45, 0.012),
                (sign * 14, 2.45, 0.012),
            ],
            white,
        )
        circle("Free throw circle", (sign * 8.2, 0, 0.012), 1.8, white)
        points = [
            (sign * (12.425 - 6.75 * math.cos(a)), 6.75 * math.sin(a), 0.012)
            for a in [-1.36 + i * 2.72 / 96 for i in range(97)]
        ]
        line(
            "Three point arc",
            [(sign * 14, points[0][1], 0.012)] + points + [(sign * 14, points[-1][1], 0.012)],
            white,
        )
        for x in (9, 10, 11, 12):
            for side in (-1, 1):
                line(
                    "Lane mark",
                    [(sign * x, side * 2.45, 0.012), (sign * x, side * 2.65, 0.012)],
                    white,
                )
    line(
        "Boundary",
        [(-14, -7.5, 0.012), (14, -7.5, 0.012), (14, 7.5, 0.012), (-14, 7.5, 0.012)],
        white,
        closed=True,
    )
    line("Half court", [(0, -7.5, 0.012), (0, 7.5, 0.012)], white)
    circle("Center circle", (0, 0, 0.012), 1.8, white)
    text_object("BASKETBALL", (0, 8.0, 0.016), 0.55, key, rotation=(0, 0, 0))
    box("Far wall", (0, 15, 5.5), (42, 0.3, 11), concrete)
    for sign in (-1, 1):
        box("End wall", (sign * 21, 0, 5.5), (0.3, 30, 11), concrete)
    seats = [material("Seat", tuple(color), 0.55) for color in hall["seat_colors"]]
    arena_style = hall.get("style", "arena") == "arena"
    for row in range(SEAT_ROWS if arena_style else 0):
        _x, y, z = seat_position(row, 0)
        box("Stand tier", (0, y, z * 0.5), (34, 0.86, z), concrete)
        for col in range(SEAT_COLUMNS):
            x, y, z = seat_position(row, col)
            mat = seats[(row + col // 5) % 2]
            box("Seat", (x, y, z + 0.26), (0.62, 0.52, 0.1), mat, bevel=0.04)
            box("Seat back", (x, y + 0.24, z + 0.53), (0.62, 0.09, 0.50), mat, bevel=0.04)
    for x in range(-18, 19, 6):
        box("Roof beam", (x, 0, 10.5), (0.18, 30, 0.35), steel)
        box("Wall column", (x, 14.6, 5), (0.26, 0.32, 10), steel)
    black = material("Scoreboard", (0.006, 0.009, 0.015), 0.7)
    box("Scoreboard", (0, 14.7, 6.1), (5, 0.16, 2.0), black, bevel=0.08)
    score = random.Random(hall["spectator_seed"])
    text_object(
        f"HOME   {score.randrange(10, 90)} : {score.randrange(10, 90)}   AWAY",
        (0, 14.58, 6.45),
        0.32,
        white,
    )
    text_object(
        f"{score.randrange(0, 10):02d}:{score.randrange(0, 60):02d}     Q{score.randrange(1, 5)}",
        (0, 14.57, 5.6),
        0.45,
        white,
    )
    rows = hall["light_rows"]
    for x in (-10, 0, 10):
        for k in range(rows):
            y = -5 + 10 * k / (rows - 1)
            data = bpy.data.lights.new("Arena softbox", "AREA")
            data.energy = hall["light_energy"] * 2 / rows
            data.use_temperature = True
            data.temperature = hall["light_kelvin"]
            data.shape, data.size, data.size_y = "RECTANGLE", 5, 2
            obj = bpy.data.objects.new("Arena softbox", data)
            bpy.context.collection.objects.link(obj)
            obj.location = (x, y, 9)
    if hall.get("windows"):
        windows(hall)
    if hall.get("wall_bars"):
        wall_bars()
    spectators(hall)
    return white, steel


def other_sport_lines(sport, mat, z=0.009):
    """Court lines of other sports in a multi-sport hall. They are floor markings only."""
    width = 0.02
    if sport == "handball":  # 40 x 20 m, 6 m goal areas and dashed 9 m lines
        line(
            "Handball boundary",
            [(-20, -10, z), (20, -10, z), (20, 10, z), (-20, 10, z)],
            mat,
            width,
            closed=True,
        )
        line("Handball centre", [(0, -10, z), (0, 10, z)], mat, width)
        for sign in (-1, 1):
            for radius, dashed in ((6.0, False), (9.0, True)):
                points = []
                for k in range(49):
                    a = -math.pi / 2 + k * math.pi / 48
                    post = 1.5 if math.sin(a) >= 0 else -1.5
                    points.append(
                        (sign * (20 - radius * math.cos(a)), post + radius * math.sin(a), z)
                    )
                if dashed:
                    for k in range(0, 48, 2):
                        line("Handball 9 m", points[k : k + 2], mat, width)
                else:
                    line("Handball 6 m", points, mat, width)
    elif sport == "volleyball":  # 18 x 9 m with attack lines
        line(
            "Volleyball boundary",
            [(-9, -4.5, z), (9, -4.5, z), (9, 4.5, z), (-9, 4.5, z)],
            mat,
            width,
            closed=True,
        )
        for x in (-3, 0, 3):
            line("Volleyball line", [(x, -4.5, z), (x, 4.5, z)], mat, width)
    elif sport == "badminton":  # three 6.1 x 13.4 m courts across the hall
        for cx in (-8, 0, 8):
            line(
                "Badminton boundary",
                [
                    (cx - 3.05, -6.7, z),
                    (cx + 3.05, -6.7, z),
                    (cx + 3.05, 6.7, z),
                    (cx - 3.05, 6.7, z),
                ],
                mat,
                width,
                closed=True,
            )
            for y in (-1.98, 0, 1.98):
                line("Badminton line", [(cx - 3.05, y, z), (cx + 3.05, y, z)], mat, width)


def windows(hall):
    """A band of daylight windows high on the far wall, and the sun shining through."""
    glass = material("Window", (0.75, 0.82, 0.9), 0.1)
    shader = glass.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Emission Color"].default_value = (0.8, 0.88, 1.0, 1)
    shader.inputs["Emission Strength"].default_value = 3.0
    frame = material("Window frame", (0.85, 0.86, 0.85), 0.5)
    for x in range(-18, 19, 3):
        box("Window", (x, 14.82, 8.4), (2.6, 0.05, 1.8), glass)
        box("Window mullion", (x + 1.5, 14.8, 8.4), (0.12, 0.08, 1.9), frame)
    sun = bpy.data.lights.new("Daylight", "SUN")
    sun.energy = 2.5
    sun.angle = 0.02
    obj = bpy.data.objects.new("Daylight", sun)
    bpy.context.collection.objects.link(obj)
    obj.rotation_euler = (hall["sun_angle"], 0, math.pi)  # from the window side, downwards


def wall_bars():
    """Wooden wall bars along the far wall, common in school sports halls."""
    wood = material("Wall bar wood", (0.6, 0.42, 0.22), 0.5)
    left, right, y = -12.0, 12.0, 14.75
    for x in [left + k for k in range(int(right - left) + 1)]:
        limb("Wall bar post", (x, y, 0.0), (x, y, 2.6), 0.035, 0.035, wood)
    for k in range(16):
        z = 0.15 + k * 0.155
        line("Wall bar rung", [(left, y - 0.02, z), (right, y - 0.02, z)], wood, 0.02)


def shared_mesh(kind):
    """Mesh data for many identical background objects, with one material slot."""
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8)
    template = bpy.context.object
    mesh = template.data
    mesh.materials.append(None)
    bpy.data.objects.remove(template)
    return mesh


def spectators(hall):
    """Seated spectators on the stands. They are background (ID 0), not labelled people."""
    if hall["spectator_density"] <= 0:
        return
    rng = random.Random(hall["spectator_seed"])
    torso_mesh, head_mesh = shared_mesh("cube"), shared_mesh("sphere")
    shirts = [
        material("Spectator shirt", tuple(rng.uniform(0.02, 0.8) for _ in range(3)), 0.8)
        for _ in range(10)
    ]
    skins = [
        material("Spectator skin", color, 0.55)
        for color in ((0.40, 0.20, 0.10), (0.18, 0.075, 0.035), (0.68, 0.40, 0.23))
    ]

    def place(name, mesh, location, scale, mat):
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.location, obj.scale, obj.pass_index = location, scale, 0
        obj.material_slots[0].link = "OBJECT"  # own colour, shared geometry
        obj.material_slots[0].material = mat

    for row in range(SEAT_ROWS):
        for col in range(SEAT_COLUMNS):
            if rng.random() >= hall["spectator_density"]:
                continue
            x, y, z = seat_position(row, col)
            x += rng.uniform(-0.08, 0.08)
            place(
                "Spectator",
                torso_mesh,
                (x, y + 0.05, z + 0.62),
                (0.42, 0.26, 0.6),
                rng.choice(shirts),
            )
            place(
                "Spectator head",
                head_mesh,
                (x, y + 0.02, z + 1.05),
                (0.1, 0.11, 0.13),
                rng.choice(skins),
            )


def hoop(sign, identity, white, steel):
    orange = material("Rim orange", (0.85, 0.16, 0.025), 0.35, 0.2)
    board = material("Opaque practice backboard", (0.80, 0.84, 0.82), 0.25)
    # Annotation policy: backboard, rim and net; support/padding are background.
    box("Backboard", (sign * 12.8, 0, 3.425), (0.07, 1.8, 1.05), board, identity, bevel=0.012)
    markings = material("Backboard markings", (0.015, 0.015, 0.015), 0.7)
    x = sign * 12.755
    line(
        "Backboard target",
        [(x, -0.295, 3.05), (x, -0.295, 3.50), (x, 0.295, 3.50), (x, 0.295, 3.05)],
        markings,
        0.022,
        identity,
        closed=True,
    )
    center = (sign * 12.425, 0, 3.05)
    circle("Basket rim", center, 0.225, orange, 0.018, identity)
    box("Rim mounting bracket", (sign * 12.72, 0, 3.05), (0.18, 0.12, 0.04), orange, identity)
    line(
        "Backboard boundary",
        [(x, -0.875, 2.925), (x, 0.875, 2.925), (x, 0.875, 3.925), (x, -0.875, 3.925)],
        markings,
        0.025,
        identity,
        closed=True,
    )
    for k in range(12):
        theta = k * math.tau / 12
        for direction in (-1, 1):
            points = []
            for row in range(5):
                angle = theta + direction * row * math.pi / 12
                radius = 0.225 - 0.02 * row
                points.append(
                    (
                        center[0] + radius * math.cos(angle),
                        radius * math.sin(angle),
                        3.03 - 0.10 * row,
                    )
                )
            line("Net cord", points, white, 0.0035, identity)
    box("Basket base", (sign * 16, 0, 0.4), (1.8, 1.25, 0.8), steel, bevel=0.10)
    limb("Basket stanchion", (sign * 16, 0, 0.8), (sign * 15, 0, 4.5), 0.12, 0.12, steel)
    limb("Backboard support", (sign * 15, 0, 4.5), (sign * 12.9, 0, 3.65), 0.10, 0.10, steel)


def jersey_mesh(mat, identity, root, drop):
    rings = [
        (0.92, 0.20, 0.13),
        (1.08, 0.20, 0.135),
        (1.34, 0.25, 0.15),
        (1.48, 0.27, 0.145),
        (1.57, 0.17, 0.11),
    ]
    vertices = [
        (math.cos(i * math.tau / 24) * rx, math.sin(i * math.tau / 24) * ry, z - drop)
        for z, rx, ry in rings
        for i in range(24)
    ]
    faces = [
        (j * 24 + i, j * 24 + (i + 1) % 24, (j + 1) * 24 + (i + 1) % 24, (j + 1) * 24 + i)
        for j in range(4)
        for i in range(24)
    ]
    mesh = bpy.data.meshes.new("Tailored jersey")
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new("Jersey", mesh)
    bpy.context.collection.objects.link(obj)
    finish(obj, "Jersey", mat, identity, root)
    mod = obj.modifiers.new("Cloth silhouette", "SUBSURF")
    mod.levels = 1
    for poly in mesh.polygons:
        poly.use_smooth = True


# OpenPose BODY_18 order, the format pose ControlNets expect.
OPENPOSE_18 = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
# instance ID -> (root object, joint name -> local position). Athletes face local -Y,
# so their right side is local -X (side == -1).
SKELETONS = {}


def along(a, b, t):
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))


# instance ID -> (upper-body pivot, first-frame crouch, per-frame updaters)
RIGS = {}


def set_limb(obj, a, b):
    """Place a limb made by limb() between two points; bone lengths never change."""
    a, b = Vector(a), Vector(b)
    obj.location = (a + b) * 0.5
    obj.rotation_euler = (b - a).to_track_quat("Z", "Y").to_euler()


def athlete(plan):
    """Build one athlete from the planned skeleton and register how it follows each frame.

    Legs, arms, joints and shoes are placed from the frame's joints; head, torso, jersey
    and lettering hang from an upper-body pivot that sinks with the frame's crouch.
    """
    identity, team = plan["instance_id"], plan["team"]
    j = {name: tuple(value) for name, value in plan["joints"].items()}
    drop = plan["pose"]["crouch"]
    root = bpy.data.objects.new("Referee" if team == 2 else f"Player {plan['number']}", None)
    bpy.context.collection.objects.link(root)
    upper = bpy.data.objects.new("Upper body", None)
    bpy.context.collection.objects.link(upper)
    upper.parent = root
    updaters = []

    def at_joint(obj, name, offset=(0.0, 0.0, 0.0)):
        updaters.append(
            lambda joints, obj=obj, name=name, offset=offset: setattr(
                obj, "location", tuple(Vector(joints[name]) + Vector(offset))
            )
        )
        return obj

    def between(obj, start, end):
        updaters.append(lambda joints, obj=obj: set_limb(obj, start(joints), end(joints)))
        return obj

    skin = material("Skin", tuple(plan["skin"]), 0.53)
    skin.node_tree.nodes.get("Principled BSDF").inputs["Subsurface Weight"].default_value = 0.08
    kit = material("Basketball fabric", tuple(plan["kit"]), 0.82)
    dark = material("Hair / shoes", (0.018, 0.012, 0.009), 0.7)
    light = 0.88 if sum(plan["kit"]) < 1.8 else 0.05  # lettering contrasts with the kit
    white = material("Socks and lettering", (light, light, light * 0.93), 0.8)
    socks = material("Socks", (0.88, 0.88, 0.82), 0.8)
    shorts = kit if team < 2 else dark
    jersey_mesh(kit, identity, upper, drop)
    neck = j["neck"]
    oval("Neck", (neck[0], neck[1], neck[2] + 0.04), (0.075, 0.075, 0.11), skin, identity, upper)
    head = j["head"]
    oval("Head", head, (0.103, 0.098, 0.143), skin, identity, upper)
    if plan["hair"] == "short":
        hair = ((head[0], head[1] + 0.013, head[2] + 0.075), (0.105, 0.094, 0.076))
        oval("Short hair", *hair, dark, identity, upper)
    elif plan["hair"] == "long":
        hair = ((head[0], head[1] + 0.03, head[2] + 0.02), (0.115, 0.11, 0.15))
        oval("Long hair", *hair, dark, identity, upper)
    oval("Nose", j["nose"], (0.020, 0.028, 0.033), skin, identity, upper)
    for prefix in ("right", "left"):
        oval("Ear", j[f"{prefix}_ear"], (0.018, 0.025, 0.040), skin, identity, upper)
        eye = j[f"{prefix}_eye"]
        oval("Eye", eye, (0.016, 0.008, 0.009), socks, identity, upper)
        oval("Iris", (eye[0], eye[1] - 0.008, eye[2]), (0.007, 0.003, 0.007), dark, identity, upper)
        hip_n, knee_n, ankle_n = f"{prefix}_hip", f"{prefix}_knee", f"{prefix}_ankle"
        hip, knee, ankle = j[hip_n], j[knee_n], j[ankle_n]

        def split(joints, hip_n=hip_n, knee_n=knee_n):
            return along(joints[hip_n], joints[knee_n], 0.45)

        def sock_top(joints, ankle_n=ankle_n, knee_n=knee_n):
            return along(joints[ankle_n], joints[knee_n], 0.4)

        def get(name):
            return lambda joints: joints[name]

        between(
            limb("Shorts leg", hip, split(j), 0.14, 0.13, shorts, identity, root), get(hip_n), split
        )
        between(
            limb("Thigh", split(j), knee, 0.105, 0.080, skin, identity, root), split, get(knee_n)
        )
        at_joint(oval("Knee", knee, (0.078, 0.078, 0.082), skin, identity, root), knee_n)
        between(
            limb("Calf", knee, ankle, 0.075, 0.045, skin, identity, root), get(knee_n), get(ankle_n)
        )
        between(
            limb("Sock", sock_top(j), ankle, 0.050, 0.047, socks, identity, root),
            sock_top,
            get(ankle_n),
        )
        shoe = oval(
            "Shoe", (ankle[0], ankle[1] - 0.060, 0.065), (0.067, 0.145, 0.065), dark, identity, root
        )
        at_joint(shoe, ankle_n, (0.0, -0.060, 0.065 - ankle[2]))
        sole = box(
            "Shoe sole",
            (ankle[0], ankle[1] - 0.06, 0.026),
            (0.13, 0.26, 0.033),
            socks,
            identity,
            root,
            0.014,
        )
        at_joint(sole, ankle_n, (0.0, -0.06, 0.026 - ankle[2]))
        shoulder_n, elbow_n, wrist_n = f"{prefix}_shoulder", f"{prefix}_elbow", f"{prefix}_wrist"
        shoulder, elbow, wrist = j[shoulder_n], j[elbow_n], j[wrist_n]
        at_joint(oval("Shoulder", shoulder, (0.098, 0.098, 0.11), skin, identity, root), shoulder_n)
        between(
            limb("Upper arm", shoulder, elbow, 0.083, 0.061, skin, identity, root),
            get(shoulder_n),
            get(elbow_n),
        )
        at_joint(oval("Elbow", elbow, (0.061, 0.061, 0.061), skin, identity, root), elbow_n)
        between(
            limb("Forearm", elbow, wrist, 0.061, 0.035, skin, identity, root),
            get(elbow_n),
            get(wrist_n),
        )

        def fingertips(joints, elbow_n=elbow_n, wrist_n=wrist_n):
            direction = Vector(joints[wrist_n]) - Vector(joints[elbow_n])
            direction.normalize()
            return tuple(Vector(joints[wrist_n]) + direction * 0.09)

        hand = limb("Hand", wrist, fingertips(j), 0.045, 0.04, skin, identity, root)
        hand.scale = (1, 0.6, 1)
        between(hand, get(wrist_n), fingertips)
    torso_z = 1.20 - drop
    if team < 2:
        text_object(
            str(plan["number"]), (0, -0.153, torso_z), 0.19, white, identity=identity, parent=upper
        )
        text_object(
            str(plan["number"]),
            (0, 0.155, torso_z),
            0.21,
            white,
            rotation=(math.pi / 2, 0, math.pi),
            identity=identity,
            parent=upper,
        )
        for side in (-1, 1):
            trim = [(side * 0.203, 0, 0.96 - drop), (side * 0.25, 0, 1.44 - drop)]
            line("Uniform trim", trim, white, 0.012, identity, upper)
    else:
        for x in (-0.15, -0.075, 0, 0.075, 0.15):
            stripe = [(x, -0.139, 1.02 - drop), (x, -0.151, 1.45 - drop)]
            line("Referee stripe", stripe, dark, 0.014, identity, upper)
    RIGS[identity] = (upper, drop, updaters)
    SKELETONS[identity] = (root, {name: j[name] for name in OPENPOSE_18})
    return root


def pose_athlete(plan, root, frame):
    """Move an athlete to a frame: root transform, crouch and every limb from the joints."""
    root.location = tuple(frame["position"])
    root.rotation_euler = (0, 0, frame["heading"])
    root.scale = (1, 1, frame["height_scale"])
    upper, first_crouch, updaters = RIGS[plan["instance_id"]]
    joints = {name: tuple(value) for name, value in frame.get("joints", plan["joints"]).items()}
    upper.location = (0, 0, first_crouch - frame.get("crouch", first_crouch))
    for update in updaters:
        update(joints)
    SKELETONS[plan["instance_id"]] = (root, {name: joints[name] for name in OPENPOSE_18})


def bench_people(people):
    """Substitutes on benches and standing coaches: background people, never labelled."""
    if not people:
        return
    seat = material("Team bench", (0.05, 0.06, 0.08), 0.6)
    for person in people:
        x, y = person["position"]
        cloth = material("Bench clothing", tuple(person["kit"]), 0.8)
        skin = material("Bench skin", tuple(person["skin"]), 0.55)
        lean = person["lean"]
        if person["kind"] == "substitute":
            hip_z = 0.5
            box("Bench seat", (x, y + 0.05, 0.43), (0.56, 0.45, 0.06), seat)
            box("Bench back", (x, y + 0.3, 0.7), (0.56, 0.05, 0.5), seat)
            for side in (-1, 1):
                knee = (x + side * 0.11, y - 0.42, 0.5)
                limb("Bench thigh", (x + side * 0.11, y, hip_z), knee, 0.09, 0.075, cloth)
                limb("Bench shin", knee, (x + side * 0.12, y - 0.48, 0.1), 0.065, 0.045, skin)
                oval("Bench shoe", (x + side * 0.12, y - 0.55, 0.06), (0.06, 0.13, 0.06), seat)
        else:
            hip_z = 0.95
            for side in (-1, 1):
                limb(
                    "Coach leg",
                    (x + side * 0.1, y, hip_z),
                    (x + side * 0.12, y, 0.08),
                    0.09,
                    0.06,
                    cloth,
                )
                oval("Coach shoe", (x + side * 0.12, y - 0.06, 0.06), (0.06, 0.13, 0.06), seat)
        forward = Vector((0, -math.sin(lean), math.cos(lean)))
        hip = Vector((x, y, hip_z))
        torso = box("Bench torso", tuple(hip + forward * 0.32), (0.4, 0.24, 0.6), cloth)
        torso.rotation_euler = (-lean, 0, 0)
        oval("Bench head", tuple(hip + forward * 0.75), (0.1, 0.1, 0.13), skin)
        for side in (-1, 1):
            shoulder = hip + forward * 0.56 + Vector((side * 0.24, 0, 0))
            hand = shoulder + Vector((side * 0.05, -0.18, -0.42))
            limb(
                "Bench arm",
                tuple(shoulder),
                tuple(hand),
                0.06,
                0.045,
                cloth if person["kind"] == "coach" else skin,
            )


def ball():
    orange = material("Basketball leather", (0.72, 0.19, 0.027), 0.78)
    nodes, links = orange.node_tree.nodes, orange.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 150
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.003
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs[0], nodes.get("Principled BSDF").inputs["Normal"])
    root = bpy.data.objects.new("Ball root", None)
    bpy.context.collection.objects.link(root)
    oval("Basketball", (0, 0, 0), (0.12,) * 3, orange, 1, root)
    seam = material("Ball seams", (0.014, 0.010, 0.008), 0.9)
    for axis in range(3):
        points = []
        for i in range(96):
            p = [0.121 * math.cos(i * math.tau / 96), 0.121 * math.sin(i * math.tau / 96)]
            p.insert(axis, 0)
            points.append(p)
        line("Ball seam", points, seam, 0.002, 1, root, True)
    return root


def mask_output(scene, output):
    scene.view_layers[0].use_pass_object_index = True
    scene.view_layers[0].pass_alpha_threshold = 0.5
    tree = bpy.data.node_groups.new("Raw instance IDs", "CompositorNodeTree")
    scene.compositing_node_group = tree
    layers = tree.nodes.new("CompositorNodeRLayers")
    target = tree.nodes.new("CompositorNodeOutputFile")
    target.directory = str(output)
    target.file_name = "object_ids"
    target.format.media_type = "IMAGE"
    target.format.file_format = "OPEN_EXR"
    target.format.color_depth = "32"
    target.save_as_render = False
    target.file_output_items.clear()
    target.file_output_items.new("FLOAT", "IDs")
    tree.links.new(layers.outputs["Object Index"], target.inputs[0])


def extract_ids(output, destination, width, height):
    import numpy as np  # bundled with Blender

    paths = list(output.glob("*.exr"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one raw Object Index pass, got {paths}")
    image = bpy.data.images.load(str(paths[0]), check_existing=False)
    if tuple(image.size) != (width, height):
        raise RuntimeError("Instance pass dimensions differ from RGB")
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    # Blender's image buffer starts at the bottom left; exported PGM at top left.
    raw = values.reshape(height, width, image.channels)[::-1, :, 0]
    rounded = np.rint(raw)
    if (
        not np.all(np.isfinite(raw))
        or rounded.min() < 0
        or rounded.max() > 255
        or np.abs(raw - rounded).max() > 0.0001
    ):
        raise RuntimeError("Non-integer or out-of-range object ID")
    ids = np.ascontiguousarray(rounded, dtype=np.uint8).tobytes()
    destination.write_bytes(f"P5\n{width} {height}\n255\n".encode() + ids)
    bpy.data.images.remove(image)
    paths[0].unlink()


# FIBA court: 28 m x 15 m.
COURT_HALF_WIDTH = 7.5
BACKBOARD_X, BACKBOARD_TOP = 12.8, 3.95  # matches hoop() geometry


def broadcast_camera(camera, rng, focus_x):
    """Elevated sideline perspective camera aimed at the action; drifts slightly per frame."""
    camera.data.type = "PERSP"
    camera.data.lens, camera.data.sensor_width = 35, 36
    focus_x = max(min(focus_x, 8.0), -8.0)
    base = (focus_x * 0.6 + rng.uniform(-1.5, 1.5), -rng.uniform(19, 22), rng.uniform(9, 11))
    target = Vector((focus_x, 0, 1.35))

    def place(frame):
        camera.location = (base[0] + frame * 0.12, base[1], base[2])
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()

    return place, {"preset": "broadcast", "projection": "perspective", "lens_mm": 35}


def reco_panorama_camera(camera, rng, width, height):
    """Approximate a stitched Reco full-court panorama.

    Reco stitches two wide-angle cameras at the centre of one sideline and addresses the
    result by yaw/pitch. This preset renders that view directly as an equirectangular
    panorama: a static, low mount just outside the near sideline, a horizontal sweep that
    covers the whole court, and square pixels (vertical sweep follows the aspect ratio).
    Mount height, setback, offset, sweep, tilt and roll are randomized per scene seed.
    """
    data = camera.data
    data.type = "PANO"
    data.panorama_type = "EQUIRECTANGULAR"
    mount_height = rng.uniform(2.0, 4.5)
    setback = rng.uniform(0.5, 2.0)
    offset = rng.uniform(-1.0, 1.0)
    sweep = math.radians(rng.uniform(160, 180))
    vertical = sweep * height / width
    if vertical > math.radians(110):
        raise RuntimeError("reco_panorama needs a wide aspect ratio, for example 2560x720")
    # Keep both backboards inside the frame with a little headroom, like a real
    # installation is aimed; the near floor below the camera is cropped instead.
    hoop_distance = math.hypot(BACKBOARD_X, COURT_HALF_WIDTH + setback)
    backboard_elevation = math.atan2(BACKBOARD_TOP - mount_height, hoop_distance)
    top = backboard_elevation + math.radians(rng.uniform(3, 10))
    data.longitude_min, data.longitude_max = -sweep / 2, sweep / 2
    data.latitude_min, data.latitude_max = top - vertical, top
    camera.location = (offset, -(COURT_HALF_WIDTH + setback), mount_height)
    # Local -Z looks along +Y (across the court), local +Y is up; small stitching roll.
    roll = math.radians(rng.uniform(-0.7, 0.7))
    camera.rotation_euler = (math.pi / 2, roll, 0)

    def place(_frame):
        pass  # Reco cameras are fixed installations.

    return place, {
        "preset": "reco_panorama",
        "projection": "equirectangular",
        "mount_height_m": mount_height,
        "sideline_setback_m": setback,
        "longitude_range_rad": [data.longitude_min, data.longitude_max],
        "latitude_range_rad": [data.latitude_min, data.latitude_max],
    }


def project_pixel(scene, camera, point, width, height):
    """Pixel-centre coordinates of a world point for perspective or equirectangular cameras."""
    data = camera.data
    if data.type != "PANO":
        uv = world_to_camera_view(scene, camera, Vector(point))
        if uv.z <= 0:
            return None  # behind the camera
        return [uv.x * width - 0.5, (1 - uv.y) * height - 0.5]
    if data.panorama_type != "EQUIRECTANGULAR":
        raise RuntimeError("Only equirectangular panoramas can be projected")
    local = camera.matrix_world.inverted() @ Vector(point)
    # Cycles convention, verified with rendered markers: forward -Z, right +X, up +Y.
    longitude = math.atan2(local.x, -local.z)
    latitude = math.atan2(local.y, math.hypot(local.x, local.z))
    u = (longitude - data.longitude_min) / (data.longitude_max - data.longitude_min)
    v = (data.latitude_max - latitude) / (data.latitude_max - data.latitude_min)
    return [u * width - 0.5, v * height - 0.5]


# Outward surface normals (athlete-local) of face keypoints and the minimum cosine to the
# camera at which each stays visible. A face point turned away is hidden by the athlete's
# own head, which the ID pass cannot reveal. Eyes look forward and slightly outward; the
# nose stays visible a little past a pure profile, as in OpenPose output.
FACE_VISIBILITY = {
    "nose": ((0, -1, 0), -0.2),
    "right_eye": ((-0.55, -0.83, 0), 0.0),
    "left_eye": ((0.55, -0.83, 0), 0.0),
    "right_ear": ((-1, 0, 0), 0.0),
    "left_ear": ((1, 0, 0), 0.0),
}


def pose_keypoints(scene, camera, root, skeleton, job):
    """Projected joints; face points carry a third value, False when self-occluded."""
    camera_position = camera.matrix_world.translation
    rotation = root.matrix_world.to_3x3().normalized()
    keypoints = []
    for name, local in skeleton.items():
        point = root.matrix_world @ Vector(local)
        pixel = project_pixel(scene, camera, point, job["width"], job["height"])
        if pixel is not None and name in FACE_VISIBILITY:
            normal, minimum = FACE_VISIBILITY[name]
            towards_camera = (camera_position - point).normalized()
            pixel.append((rotation @ Vector(normal)).normalized().dot(towards_camera) > minimum)
        keypoints.append(pixel)
    return keypoints


def main():
    if bpy.app.version[0] != 5:
        raise RuntimeError("This basketball backend requires Blender 5.x")
    request = Path(sys.argv[sys.argv.index("--") + 1])
    settings = json.loads(request.read_text())
    job, output = settings["job"], request.parent
    rng = random.Random(job["seed"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    device = "CPU"
    if settings.get("device") == "auto" and sys.platform == "darwin":
        try:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = "METAL"
            preferences.get_devices()
            available = [item for item in preferences.devices if item.type == "METAL"]
            if available:
                for item in preferences.devices:
                    item.use = item.type == "METAL"
                scene.cycles.device = "GPU"
                device = "METAL"
        except (TypeError, RuntimeError):
            pass
    scene.cycles.samples = settings["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.seed = job["seed"] % 2147483647
    scene.cycles.use_animated_seed = False
    scene.render.resolution_x, scene.render.resolution_y = job["width"], job["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "AgX"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes.get("Background").inputs["Color"].default_value = (
        0.55,
        0.66,
        0.82,
        1,
    )
    scene.world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 0.32
    layout = settings["layout"]
    white, steel = arena(layout["hall"])
    hoop(-1, 40, white, steel)
    hoop(1, 41, white, steel)
    athletes = [(plan, athlete(plan)) for plan in layout["athletes"]]
    bench_people(layout.get("benches", []))
    ball_root = ball()
    preset = job.get("camera", "broadcast")
    camera_data = bpy.data.cameras.new("Dataset camera")
    camera = bpy.data.objects.new("Dataset camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    if preset == "broadcast":
        place_camera, camera_info = broadcast_camera(camera, rng, layout["focus_x"])
    elif preset == "reco_panorama":
        place_camera, camera_info = reco_panorama_camera(camera, rng, job["width"], job["height"])
    else:
        raise RuntimeError(f"Unknown camera preset {preset!r}")
    mask_dir = output / "id-pass"
    mask_dir.mkdir()
    mask_output(scene, mask_dir)
    classes = {"ball": 0, "player": 1, "referee": 2, "hoop": 3}
    identities = [{"instance_id": 1, "class_id": 0, "label": "ball"}]
    identities += [
        {
            "instance_id": plan["instance_id"],
            "class_id": classes[plan["label"]],
            "label": plan["label"],
        }
        for plan in layout["athletes"]
    ]
    identities += [{"instance_id": i, "class_id": 3, "label": "hoop"} for i in (40, 41)]
    frames, cameras = [], []
    started = time.monotonic()
    for frame in range(settings["frames"]):
        scene.frame_set(frame + 1)
        place_camera(frame)
        for plan, root in athletes:
            pose_athlete(plan, root, plan["frames"][frame])
        ball_root.location = tuple(layout["ball"]["positions"][frame])
        bpy.context.view_layer.update()
        stem = f"frame-{frame:06d}"
        scene.render.filepath = str(output / (stem + ".png"))
        bpy.ops.render.render(write_still=True)
        extract_ids(mask_dir, output / (stem + ".instances.pgm"), job["width"], job["height"])
        frames.append(
            {
                "frame_index": frame,
                "rgb": stem + ".png",
                "instance_ids": stem + ".instances.pgm",
                "objects": identities,
                "poses": [
                    {
                        "instance_id": identity,
                        "keypoints": pose_keypoints(scene, camera, root, skeleton, job),
                    }
                    for identity, (root, skeleton) in sorted(SKELETONS.items())
                ],
            }
        )
        cameras.append(
            {
                "frame_index": frame,
                **camera_info,
                "world_matrix": [list(row) for row in camera.matrix_world],
                "sensor_width_mm": camera_data.sensor_width,
                "ball_pixel": project_pixel(
                    scene, camera, ball_root.matrix_world.translation, job["width"], job["height"]
                ),
            }
        )
        print(f"SPORTS_FRAME {frame + 1}/{settings['frames']}", flush=True)
    (output / "worker-result.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "frames": frames,
                "blender_version": bpy.app.version_string,
                "engine": "CYCLES",
                "device": device,
                "samples": settings["samples"],
                "cameras": cameras,
                "elapsed_seconds": time.monotonic() - started,
                "annotation_pass": "Object Index; no antialiasing",
                "situation": layout["situation"],
                "ball_event": layout["ball"]["event"],
                "limitations": [
                    "Procedural athletes; realism and training benefit not validated",
                    "Procedural motion model; no motion capture",
                    "Opaque practice backboards",
                ],
            },
            indent=2,
        )
    )
    mask_dir.rmdir()


if __name__ == "__main__":
    main()
