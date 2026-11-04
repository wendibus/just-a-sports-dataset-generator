# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated offline Cycles CPU worker for a calibrated natural football venue."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def material(name, color, rough=0.6, metal=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Roughness"].default_value = rough
    node.inputs["Metallic"].default_value = metal
    return mat


def turf(assets, environment_rotation=0.5):
    mat = material("Photographic lawn material", (0.08, 0.16, 0.035), 0.85)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    coords = nodes.new("ShaderNodeNewGeometry")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = 1 / 1.4
    links.new(coords.outputs["Position"], scale.inputs[0])
    images = {}
    for kind in ("Color", "NormalGL", "Roughness"):
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(assets / f"Grass004_1K-JPG_{kind}.jpg"))
        tex.extension = "REPEAT"
        if kind != "Color":
            tex.image.colorspace_settings.name = "Non-Color"
        links.new(scale.outputs[0], tex.inputs["Vector"])
        images[kind] = tex
    saturation = nodes.new("ShaderNodeHueSaturation")
    saturation.inputs["Saturation"].default_value = 1.05
    saturation.inputs["Value"].default_value = 1.12
    links.new(images["Color"].outputs[0], saturation.inputs["Color"])
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Position"], separate.inputs[0])
    stripe = nodes.new("ShaderNodeMath")
    stripe.operation = "DIVIDE"
    stripe.inputs[1].default_value = 68 / 14
    links.new(separate.outputs["Y"], stripe.inputs[0])
    floor = nodes.new("ShaderNodeMath")
    floor.operation = "FLOOR"
    links.new(stripe.outputs[0], floor.inputs[0])
    modulo = nodes.new("ShaderNodeMath")
    modulo.operation = "MODULO"
    modulo.inputs[1].default_value = 2
    links.new(floor.outputs[0], modulo.inputs[0])
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.inputs["To Min"].default_value = 0.92
    ramp.inputs["To Max"].default_value = 1.04
    links.new(modulo.outputs[0], ramp.inputs[0])
    inside = None
    for component, operation, value in (
        ("X", "GREATER_THAN", 0.0),
        ("X", "LESS_THAN", 105.0),
        ("Y", "GREATER_THAN", 0.0),
        ("Y", "LESS_THAN", 68.0),
    ):
        compare = nodes.new("ShaderNodeMath")
        compare.operation = operation
        compare.inputs[1].default_value = value
        links.new(separate.outputs[component], compare.inputs[0])
        if inside is None:
            inside = compare.outputs[0]
        else:
            conjunction = nodes.new("ShaderNodeMath")
            conjunction.operation = "MULTIPLY"
            links.new(inside, conjunction.inputs[0])
            links.new(compare.outputs[0], conjunction.inputs[1])
            inside = conjunction.outputs[0]
    field_only = nodes.new("ShaderNodeMixRGB")
    field_only.inputs[1].default_value = (1, 1, 1, 1)
    links.new(inside, field_only.inputs[0])
    links.new(ramp.outputs[0], field_only.inputs[2])
    multiply = nodes.new("ShaderNodeMixRGB")
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 1
    links.new(saturation.outputs[0], multiply.inputs[1])
    links.new(field_only.outputs[0], multiply.inputs[2])
    variation = nodes.new("ShaderNodeTexNoise")
    variation.inputs["Scale"].default_value = 0.16
    variation.inputs["Detail"].default_value = 3
    links.new(coords.outputs["Position"], variation.inputs["Vector"])
    shade = nodes.new("ShaderNodeMapRange")
    shade.inputs["To Min"].default_value = 0.78
    shade.inputs["To Max"].default_value = 1.12
    links.new(variation.outputs["Fac"], shade.inputs[0])
    varied = nodes.new("ShaderNodeMixRGB")
    varied.blend_type = "MULTIPLY"
    varied.inputs[0].default_value = 1
    links.new(multiply.outputs[0], varied.inputs[1])
    links.new(shade.outputs[0], varied.inputs[2])
    links.new(varied.outputs[0], shader.inputs["Base Color"])
    # Camera-projected photographic grass retains the real small-scale surface
    # detail. A diffuse component receives the physical goal shading.
    projected = nodes.new("ShaderNodeTexEnvironment")
    projected.image = bpy.data.images.get("approaching_storm.jpg")
    invert = nodes.new("ShaderNodeVectorMath")
    invert.operation = "SCALE"
    invert.inputs[3].default_value = -1
    links.new(coords.outputs["Incoming"], invert.inputs[0])
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = environment_rotation
    links.new(invert.outputs[0], mapping.inputs[0])
    links.new(mapping.outputs[0], projected.inputs[0])
    photographic = nodes.new("ShaderNodeMixRGB")
    photographic.blend_type = "MULTIPLY"
    photographic.inputs[0].default_value = 1
    links.new(projected.outputs[0], photographic.inputs[1])
    links.new(field_only.outputs[0], photographic.inputs[2])
    emission = nodes.new("ShaderNodeEmission")
    links.new(photographic.outputs[0], emission.inputs[0])
    shaded = nodes.new("ShaderNodeMixShader")
    shaded.inputs[0].default_value = 0.12
    links.new(emission.outputs[0], shaded.inputs[1])
    links.new(shader.outputs[0], shaded.inputs[2])
    normal = nodes.new("ShaderNodeNormalMap")
    normal.inputs["Strength"].default_value = 0.12
    links.new(images["NormalGL"].outputs[0], normal.inputs["Color"])
    links.new(normal.outputs[0], shader.inputs["Normal"])
    shader.inputs["Roughness"].default_value = 0.95
    shader.inputs["Specular IOR Level"].default_value = 0.0
    # Fade the synthetic lawn into the CC0 photographic surroundings outside
    # the marked field, so a perfectly straight artificial horizon is impossible.
    edge = None
    for component, operation, value in (
        ("X", "ADD", 6.0),
        ("X", "SUBTRACT", 111.0),
        ("Y", "ADD", 6.0),
        ("Y", "SUBTRACT", 74.0),
    ):
        distance = nodes.new("ShaderNodeMath")
        distance.operation = operation
        if operation == "ADD":
            links.new(separate.outputs[component], distance.inputs[0])
            distance.inputs[1].default_value = value
        else:
            distance.inputs[0].default_value = value
            links.new(separate.outputs[component], distance.inputs[1])
        if edge is None:
            edge = distance.outputs[0]
        else:
            minimum = nodes.new("ShaderNodeMath")
            minimum.operation = "MINIMUM"
            links.new(edge, minimum.inputs[0])
            links.new(distance.outputs[0], minimum.inputs[1])
            edge = minimum.outputs[0]
    fade = nodes.new("ShaderNodeMapRange")
    fade.clamp = True
    fade.inputs["From Min"].default_value = 0.0
    fade.inputs["From Max"].default_value = 6.0
    links.new(edge, fade.inputs[0])
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(fade.outputs[0], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(shaded.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes.get("Material Output").inputs[0])
    return mat


def curve(name, points, radius, mat, identity=0):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 1
    data.bevel_depth = radius
    data.bevel_resolution = 1
    spline = data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, xyz in zip(spline.points, points):
        point.co = (*xyz, 1)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.pass_index = identity
    return obj


def marking(points, paint, width=0.12):
    verts = []
    faces = []
    for a, b in itertools.pairwise(points):
        delta = Vector((b[0] - a[0], b[1] - a[1]))
        normal = Vector((-delta.y, delta.x)).normalized() * width / 2
        start = len(verts)
        verts.extend(
            [
                (a[0] + normal.x, a[1] + normal.y, 0.006),
                (a[0] - normal.x, a[1] - normal.y, 0.006),
                (b[0] - normal.x, b[1] - normal.y, 0.006),
                (b[0] + normal.x, b[1] + normal.y, 0.006),
            ]
        )
        faces.append((start, start + 1, start + 2, start + 3))
    mesh = bpy.data.meshes.new("chalk geometry")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("painted grass line", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(paint)


def setup(request):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = request.get("samples", 24)
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 2
    scene.cycles.seed = 20261104
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.render.resolution_x = 3200
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = request.get("percentage", 100)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    assets = Path(request["assets"])
    scene.world = bpy.data.worlds.new("Real outdoor light")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(assets / "approaching_storm_2k.hdr"))
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = request.get("environment_rotation", 0.5)
    links.new(coords.outputs["Generated"], mapping.inputs[0])
    links.new(mapping.outputs[0], env.inputs[0])
    links.new(env.outputs[0], nodes.get("Background").inputs[0])
    nodes.get("Background").inputs[1].default_value = 0.8
    # A higher resolution CC0 photograph is used for camera rays; the HDRI
    # remains the source of physical illumination and goal reflections.
    photo = nodes.new("ShaderNodeTexEnvironment")
    photo.image = bpy.data.images.load(str(assets / "approaching_storm.jpg"))
    links.new(mapping.outputs[0], photo.inputs[0])
    photo_bg = nodes.new("ShaderNodeBackground")
    links.new(photo.outputs[0], photo_bg.inputs[0])
    paths = nodes.new("ShaderNodeLightPath")
    split = nodes.new("ShaderNodeMixShader")
    links.new(paths.outputs["Is Camera Ray"], split.inputs[0])
    links.new(nodes.get("Background").outputs[0], split.inputs[1])
    links.new(photo_bg.outputs[0], split.inputs[2])
    links.new(split.outputs[0], nodes.get("World Output").inputs[0])
    bpy.ops.mesh.primitive_plane_add(size=300, location=(52.5, -30, 0))
    bpy.context.object.name = "Natural turf"
    bpy.context.object.data.materials.append(turf(assets, request.get("environment_rotation", 0.5)))
    paint = material("Weathered painted markings", (0.88, 0.91, 0.86), 0.95)
    noise = paint.node_tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 160
    bump = paint.node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.008
    paint.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    paint.node_tree.links.new(
        bump.outputs[0], paint.node_tree.nodes.get("Principled BSDF").inputs["Normal"]
    )
    marking([(0, 0), (105, 0), (105, 68), (0, 68), (0, 0)], paint)
    marking([(52.5, 0), (52.5, 68)], paint)
    marking(
        [
            (52.5 + 9.15 * math.cos(i * math.tau / 160), 34 + 9.15 * math.sin(i * math.tau / 160))
            for i in range(161)
        ],
        paint,
    )
    for end, direction in ((0, 1), (105, -1)):
        for length, width in ((16.5, 40.32), (5.5, 18.32)):
            marking(
                [
                    (end, 34 - width / 2),
                    (end + direction * length, 34 - width / 2),
                    (end + direction * length, 34 + width / 2),
                    (end, 34 + width / 2),
                ],
                paint,
            )
        for x, y in ((end + direction * 11, 34), (52.5, 34)):
            bpy.ops.mesh.primitive_circle_add(
                vertices=24, radius=0.11, fill_type="NGON", location=(x, y, 0.007)
            )
            bpy.context.object.data.materials.append(paint)
    # Photograph-based surroundings remain visible outside the pitch.
    # Low wire perimeter outside the field, with an empty technical area.
    steel = material("Distant wire and rail", (0.13, 0.16, 0.14), 0.7, 0.2)
    for y in (77.0,):
        for x in range(-15, 121, 7):
            curve("fence post", [(x, y, 0), (x, y, 1.25)], 0.025, steel)
        for z in (0.18, 0.65, 1.25):
            curve("fence wire", [(-15, y, z), (120, y, z)], 0.008, steel)
    metal = material("White goal frame", (0.9, 0.92, 0.9), 0.55, 0.0)
    net = material("Woven goal net", (0.48, 0.53, 0.46), 0.84)
    goals = []
    for side, end, direction in (("left", 0, -1), ("right", 105, 1)):
        identity = 200 if side == "left" else 201
        back = end + direction * 1.8
        for y in (30.29, 37.71):
            goals.append(
                curve("goal post", [(end, y, 0.05), (end, y, 2.49)], 0.05, metal, identity)
            )
            goals.append(
                curve(
                    "goal ground frame",
                    [(end, y, 0.04), (back, y, 0.04), (back, y, 2.49)],
                    0.025,
                    metal,
                    identity,
                )
            )
            goals.append(
                curve("net roof support", [(end, y, 2.49), (back, y, 2.49)], 0.018, metal, identity)
            )
        goals.append(
            curve("goal crossbar", [(end, 30.29, 2.49), (end, 37.71, 2.49)], 0.05, metal, identity)
        )
        for i in range(43):
            y = 30.29 + i * (7.42 / 42)
            goals.append(
                curve(
                    "net upright",
                    [(back, y, 0.04), (back, y, 2.46), (end, y, 2.46)],
                    0.007,
                    net,
                    identity,
                )
            )
        for i in range(15):
            z = 0.04 + i * (2.42 / 14)
            goals.append(
                curve(
                    "net horizontal",
                    [(end, 30.29, z), (back, 30.29, z), (back, 37.71, z), (end, 37.71, z)],
                    0.007,
                    net,
                    identity,
                )
            )
        for i in range(1, 10):
            x = end + direction * i * 0.18
            for y in (30.29, 37.71):
                goals.append(
                    curve("net side upright", [(x, y, 0.04), (x, y, 2.46)], 0.007, net, identity)
                )
    data = bpy.data.cameras.new("Calibrated football camera")
    camera = bpy.data.objects.new("Calibrated football camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = (52.5, -70, 22)
    target = Vector((52.5, 34, 0))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36
    data.lens = 22.5
    data.shift_y = 210 / 3200
    data.clip_end = 5000
    bpy.context.view_layer.update()
    tests = []
    for world in ((0, 0, 0), (105, 68, 0), (52.5, 34, 0)):
        p = world_to_camera_view(scene, camera, Vector(world))
        tests.append({"world": world, "pixel": [p.x * 3200, (1 - p.y) * 900]})
    return scene, goals, tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("request", type=Path)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    request = json.loads(a.request.read_text())
    out = Path(request["output"])
    out.mkdir(parents=True, exist_ok=True)
    scene, goals, tests = setup(request)
    (out / "camera-check.json").write_text(json.dumps(tests, indent=2))
    for obj in goals:
        obj.hide_render = True
    scene.render.filepath = str(out / "without-goals.png")
    bpy.ops.render.render(write_still=True)
    for obj in goals:
        obj.hide_render = False
    if request.get("percentage", 100) == 100:
        worker = Path(__file__).with_name("basketball.py")
        spec = importlib.util.spec_from_file_location("isolated_basketball_pass_helpers", worker)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        passdir = out / "pass"
        passdir.mkdir(exist_ok=True)
        module.mask_output(scene, passdir)
    scene.render.filepath = str(out / "with-goals.png")
    bpy.ops.render.render(write_still=True)
    if request.get("percentage", 100) == 100:
        module.extract_ids(passdir, out / "goal-ids.pgm", 3200, 900)
    print("FOOTBALL VENUE COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
