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
from array import array
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


def wood_material():
    mat = material("Maple varnish, procedural grain", (0.65, 0.39, 0.16), 0.29)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Coat Weight"].default_value = 0.22
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
    ramp.color_ramp.elements[0].color = (0.28, 0.13, 0.045, 1)
    ramp.color_ramp.elements[1].color = (0.78, 0.56, 0.30, 1)
    links.new(noise.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs[0], shader.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.008
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs[0], shader.inputs["Normal"])
    return mat


def arena(rng):
    white = material("Court paint ivory", (0.90, 0.91, 0.86), 0.38)
    navy = material("Home navy paint", (0.018, 0.055, 0.10), 0.38)
    concrete = material("Warm concrete", (0.28, 0.30, 0.32), 0.8)
    steel = material("Powder coated steel", (0.08, 0.10, 0.13), 0.5, 0.35)
    oak = wood_material()
    box("Wood court", (0, 0, -0.11), (34, 22, 0.2), oak)
    # Fine plank seams, all coplanar and outside the annotation classes.
    seam = material("Fine wood joints", (0.23, 0.12, 0.05), 0.8)
    for row in range(55):
        y = -10.8 + row * 0.4
        line("Maple board joint", [(-17, y, -0.008), (17, y, -0.008)], seam, 0.003)
        for col in range(9):
            x = -16 + col * 4 + (row % 3) * 1.2
            line("Staggered board end", [(x, y, -0.007), (x, y + 0.4, -0.007)], seam, 0.002)
    for sign in (-1, 1):
        box("Painted key", (sign * 11.1, 0, 0.001), (5.8, 4.9, 0.004), navy)
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
    text_object("BASKETBALL", (0, 8.0, 0.016), 0.55, navy, rotation=(0, 0, 0))
    box("Far wall", (0, 15, 5.5), (42, 0.3, 11), concrete)
    for sign in (-1, 1):
        box("End wall", (sign * 21, 0, 5.5), (0.3, 30, 11), concrete)
    seats = [
        material("Seat blue", (0.035, 0.12, 0.22), 0.5),
        material("Seat pale", (0.46, 0.49, 0.51), 0.65),
    ]
    for row in range(4):
        y, z = 10.4 + row * 0.82, 0.30 + row * 0.43
        box("Stand tier", (0, y, z * 0.5), (34, 0.86, z), concrete)
        for col in range(42):
            x = -16.4 + col * 0.8
            mat = seats[(row + col // 5) % 2]
            box("Seat", (x, y, z + 0.26), (0.62, 0.52, 0.1), mat, bevel=0.04)
            box("Seat back", (x, y + 0.24, z + 0.53), (0.62, 0.09, 0.50), mat, bevel=0.04)
    for x in range(-18, 19, 6):
        box("Roof beam", (x, 0, 10.5), (0.18, 30, 0.35), steel)
        box("Wall column", (x, 14.6, 5), (0.26, 0.32, 10), steel)
    black = material("Scoreboard", (0.006, 0.009, 0.015), 0.7)
    box("Scoreboard", (0, 14.7, 6.1), (5, 0.16, 2.0), black, bevel=0.08)
    text_object("HOME   48 : 42   AWAY", (0, 14.58, 6.45), 0.32, white)
    text_object("03:24     Q3", (0, 14.57, 5.6), 0.45, white)
    for x in (-10, 0, 10):
        for y in (-5, 5):
            data = bpy.data.lights.new("Arena softbox", "AREA")
            data.energy = rng.uniform(1600, 2000)
            data.shape, data.size, data.size_y = "RECTANGLE", 5, 2
            obj = bpy.data.objects.new("Arena softbox", data)
            bpy.context.collection.objects.link(obj)
            obj.location = (x, y, 9)
    return white, navy, steel


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


def jersey_mesh(mat, identity, root):
    rings = [
        (0.92, 0.20, 0.13),
        (1.08, 0.20, 0.135),
        (1.34, 0.25, 0.15),
        (1.48, 0.27, 0.145),
        (1.57, 0.17, 0.11),
    ]
    vertices = [
        (math.cos(i * math.tau / 24) * rx, math.sin(i * math.tau / 24) * ry, z)
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


def athlete(identity, team, position, heading, pose, rng, number):
    root = bpy.data.objects.new("Referee" if team == 2 else f"Player {number}", None)
    bpy.context.collection.objects.link(root)
    root.location, root.rotation_euler.z = position, heading
    root.scale = (1, 1, rng.uniform(0.97, 1.10))
    skins = [(0.40, 0.20, 0.10), (0.18, 0.075, 0.035), (0.68, 0.40, 0.23), (0.33, 0.15, 0.07)]
    skin = material("Skin", rng.choice(skins), 0.53)
    skin.node_tree.nodes.get("Principled BSDF").inputs["Subsurface Weight"].default_value = 0.08
    colors = [(0.025, 0.13, 0.62), (0.86, 0.09, 0.035), (0.50, 0.52, 0.53)]
    kit = material("Basketball fabric", colors[team], 0.82)
    dark = material("Hair / shoes", (0.018, 0.012, 0.009), 0.7)
    white = material("Socks and jersey lettering", (0.88, 0.88, 0.82), 0.8)
    shorts = kit if team < 2 else dark
    jersey_mesh(kit, identity, root)
    oval("Neck", (0, 0, 1.62), (0.075, 0.075, 0.11), skin, identity, root)
    oval("Head", (0, -0.005, 1.80), (0.103, 0.098, 0.143), skin, identity, root)
    oval("Short hair", (0, 0.008, 1.875), (0.105, 0.094, 0.076), dark, identity, root)
    oval("Nose", (0, -0.102, 1.80), (0.020, 0.028, 0.033), skin, identity, root)
    for side in (-1, 1):
        oval("Ear", (side * 0.102, 0, 1.79), (0.018, 0.025, 0.040), skin, identity, root)
        oval("Eye", (side * 0.040, -0.092, 1.83), (0.016, 0.008, 0.009), white, identity, root)
        oval("Iris", (side * 0.040, -0.100, 1.83), (0.007, 0.003, 0.007), dark, identity, root)
        stride = (0.16 if side == 1 else -0.17) * (1 if pose != "defend" else 0.5)
        hip = (side * 0.12, 0, 0.97)
        knee = (side * (0.20 if pose == "defend" else 0.14), stride, 0.52)
        ankle = (side * (0.31 if pose == "defend" else 0.18), -stride, 0.13)
        limb(
            "Shorts leg",
            hip,
            (side * 0.16, stride * 0.4, 0.73),
            0.14,
            0.145,
            shorts,
            identity,
            root,
        )
        limb("Thigh", (side * 0.16, stride * 0.4, 0.75), knee, 0.11, 0.080, skin, identity, root)
        oval("Knee", knee, (0.078, 0.078, 0.082), skin, identity, root)
        limb("Calf", knee, ankle, 0.075, 0.045, skin, identity, root)
        limb("Sock", (ankle[0], ankle[1], 0.28), ankle, 0.050, 0.047, white, identity, root)
        oval(
            "Shoe", (ankle[0], ankle[1] - 0.060, 0.065), (0.067, 0.145, 0.065), dark, identity, root
        )
        box(
            "Shoe sole",
            (ankle[0], ankle[1] - 0.06, 0.026),
            (0.13, 0.26, 0.033),
            white,
            identity,
            root,
            0.014,
        )
        shoulder = (side * 0.25, 0, 1.48)
        if pose == "defend":
            elbow, hand = (side * 0.49, -0.04, 1.40), (side * 0.62, -0.18, 1.65)
        elif pose == "shoot":
            elbow, hand = (side * 0.25, -0.17, 1.75), (side * 0.10, -0.27, 2.02)
        else:
            elbow, hand = (side * 0.36, -0.09, 1.18), (side * 0.49, -0.31, 0.87)
        oval("Shoulder", shoulder, (0.098, 0.098, 0.11), skin, identity, root)
        limb("Upper arm", shoulder, elbow, 0.083, 0.061, skin, identity, root)
        oval("Elbow", elbow, (0.061, 0.061, 0.061), skin, identity, root)
        limb("Forearm", elbow, hand, 0.061, 0.035, skin, identity, root)
        oval("Hand", hand, (0.050, 0.027, 0.077), skin, identity, root)
        for finger in range(4):
            start = (hand[0] + (finger - 1.5) * 0.019, hand[1], hand[2] - 0.045)
            end = (start[0], start[1] - 0.012, start[2] - 0.055)
            limb("Finger", start, end, 0.009, 0.007, skin, identity, root)
    if team < 2:
        text_object(str(number), (0, -0.153, 1.20), 0.19, white, identity=identity, parent=root)
        text_object(
            str(number),
            (0, 0.155, 1.20),
            0.21,
            white,
            rotation=(math.pi / 2, 0, math.pi),
            identity=identity,
            parent=root,
        )
        # Narrow side panels tie upper and lower uniform together.
        for side in (-1, 1):
            line(
                "Uniform trim",
                [(side * 0.203, 0, 0.96), (side * 0.25, 0, 1.44)],
                white,
                0.012,
                identity,
                root,
            )
    else:
        for x in (-0.15, -0.075, 0, 0.075, 0.15):
            line(
                "Referee stripe",
                [(x, -0.139, 1.02), (x, -0.151, 1.45)],
                dark,
                0.014,
                identity,
                root,
            )
    return root


def ball(position):
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
    root.location = position
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
    paths = list(output.glob("*.exr"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one raw Object Index pass, got {paths}")
    image = bpy.data.images.load(str(paths[0]), check_existing=False)
    values = array("f", [0]) * len(image.pixels)
    image.pixels.foreach_get(values)
    channels = image.channels
    if tuple(image.size) != (width, height):
        raise RuntimeError("Instance pass dimensions differ from RGB")
    ids = bytearray(width * height)
    # Blender's image buffer starts at the bottom left; exported PNG/PGM at top left.
    for y in range(height):
        for x in range(width):
            value = values[((height - 1 - y) * width + x) * channels]
            rounded = round(value)
            if not 0 <= rounded <= 255 or abs(value - rounded) > 0.0001:
                raise RuntimeError("Non-integer or out-of-range object ID")
            ids[y * width + x] = rounded
    destination.write_bytes(f"P5\n{width} {height}\n255\n".encode() + ids)
    bpy.data.images.remove(image)
    paths[0].unlink()


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
    white, _navy, steel = arena(rng)
    hoop(-1, 40, white, steel)
    hoop(1, 41, white, steel)
    offense = [(3.5, -2.8), (8.0, -4.8), (10.0, 2.8), (6.0, 4.6), (1.0, 1.5)]
    defense = [(5.0, -2.0), (8.9, -3.9), (10.6, 1.5), (7.0, 3.7), (3.5, 1.7)]
    players = []
    mirrored = -1 if job["scene_index"] % 2 else 1
    for index, (x, y) in enumerate(offense + defense):
        position = (mirrored * (x + rng.uniform(-0.35, 0.35)), y + rng.uniform(-0.35, 0.35), 0)
        heading = rng.uniform(-0.4, 0.4) + (math.pi / 2 if mirrored == 1 else -math.pi / 2)
        players.append(
            athlete(
                10 + index,
                index // 5,
                position,
                heading,
                "dribble" if index == 0 else "defend" if index >= 5 else "run",
                rng,
                (index % 5) * 3 + 4,
            )
        )
    athlete(30, 2, (mirrored * 3, -5.8, 0), math.pi / 2, "run", rng, 0)
    bpy.context.view_layer.update()
    ball_root = ball(players[0].matrix_world @ Vector((-mirrored * 0.49, -0.31, 0.50)))
    camera_data = bpy.data.cameras.new("Broadcast camera")
    camera = bpy.data.objects.new("Broadcast camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens, camera_data.sensor_width = 35, 36
    mask_dir = output / "id-pass"
    mask_dir.mkdir()
    mask_output(scene, mask_dir)
    identities = [{"instance_id": 1, "class_id": 0, "label": "ball"}]
    identities += [{"instance_id": 10 + i, "class_id": 1, "label": "player"} for i in range(10)]
    identities += [{"instance_id": 30, "class_id": 2, "label": "referee"}]
    identities += [{"instance_id": i, "class_id": 3, "label": "hoop"} for i in (40, 41)]
    frames, cameras = [], []
    started = time.monotonic()
    base_camera = (mirrored * rng.uniform(-1, 2), -rng.uniform(19, 22), rng.uniform(9, 11))
    for frame in range(settings["frames"]):
        scene.frame_set(frame + 1)
        camera.location = (base_camera[0] + frame * 0.12, base_camera[1], base_camera[2])
        target = Vector((mirrored * 5, 0, 1.35))
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        ball_root.location.z += 0.045 if frame % 2 else 0
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
            }
        )
        ball_uv = world_to_camera_view(scene, camera, ball_root.location)
        cameras.append(
            {
                "frame_index": frame,
                "world_matrix": [list(row) for row in camera.matrix_world],
                "lens_mm": camera_data.lens,
                "sensor_width_mm": camera_data.sensor_width,
                "ball_pixel": [
                    ball_uv.x * job["width"] - 0.5,
                    (1 - ball_uv.y) * job["height"] - 0.5,
                ],
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
                "limitations": [
                    "Procedural athletes; realism and training benefit not validated",
                    "Static poses; no motion capture",
                    "Opaque practice backboards",
                ],
            },
            indent=2,
        )
    )
    mask_dir.rmdir()


if __name__ == "__main__":
    main()
