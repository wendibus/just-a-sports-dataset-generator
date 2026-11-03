# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated Blender worker: original procedural 3 x 2 m Futsal goals, no assets."""

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def material(name, color, roughness=0.4):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    node = m.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Roughness"].default_value = roughness
    return m


def rod(a, b, radius, mat, vertices=12):
    a, b = Vector(a), Vector(b)
    delta = b - a
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=delta.length, location=(a + b) / 2
    )
    obj = bpy.context.object
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    white = material("painted ivory metal", (0.80, 0.79, 0.74))
    red = material("red safety bands", (0.50, 0.025, 0.015))
    net = material("woven off-white cord", (0.64, 0.63, 0.58), 0.8)
    # Inside aperture y +/-1.5, z 0..2; posts extend outward by 8 cm.
    for y in (-1.54, 1.54):
        for i in range(10):
            rod((0, y, i * 0.208), (0, y, (i + 1) * 0.208), 0.04, red if i % 2 == 0 else white)
        rod((0, y, 0.04), (-1, y, 0.04), 0.018, white)
        rod((-1, y, 0.04), (-0.8, y, 1.95), 0.018, white)
        rod((0, y, 2.04), (-0.8, y, 1.95), 0.018, white)
    for i in range(16):
        rod(
            (0, -1.58 + i * 0.1975, 2.04),
            (0, -1.58 + (i + 1) * 0.1975, 2.04),
            0.04,
            red if i % 2 else white,
        )
    # Square netting, 10 cm mesh. Holes stay transparent in RGB and IDs.
    for j in range(32):
        y = -1.55 + j * 0.1
        rod((-1, y, 0.04), (-0.8, y, 2.0), 0.004, net, 6)
        rod((-0.8, y, 2.0), (0, y, 2.04), 0.004, net, 6)
    for k in range(21):
        z = 0.04 + k * 0.098
        x = -1 + 0.2 * k / 20
        rod((x, -1.54, z), (x, 1.54, z), 0.004, net, 6)
        for y in (-1.54, 1.54):
            rod((x, y, z), (0, y, z), 0.004, net, 6)
    for j in range(1, 10):
        x = -1 + j * 0.1
        for y in (-1.54, 1.54):
            rod((x, y, 0.04), (x * 0.8, y, 2.0), 0.004, net, 6)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 850
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("soft gym ambient")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.68, 0.72, 0.75, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.65
    for loc, power, size in [((2, 0, 6), 650, 6), ((-3, -4, 4), 450, 5)]:
        data = bpy.data.lights.new("diffuse window", "AREA")
        data.energy = power
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new("window fill", data)
        scene.collection.objects.link(obj)
        obj.location = loc
        obj.rotation_euler = (Vector((0, 0, 1)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    data.type = "ORTHO"
    data.ortho_scale = 4.5
    records = []
    for side, y in [("left", 5.0), ("right", -5.0)]:
        camera.location = (8, y, 3.3)
        camera.rotation_euler = (
            (Vector((-0.25, 0, 1)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        scene.render.filepath = str(args.output / f"goal-{side}.png")
        bpy.ops.render.render(write_still=True)
        records.append(
            {
                "id": f"goal-{side}",
                "file": f"goals/goal-{side}.png",
                "opening_m": [3, 2],
                "depth_m": 1,
                "source": "Original procedural Blender geometry",
                "rights": "Original project geometry; rendered output may be used with the generated dataset",
            }
        )
    (args.output / "geometry.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
