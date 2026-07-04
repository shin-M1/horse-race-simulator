from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def render_with_blender(payload: dict[str, Any], output_path: str) -> str:
    """Render a simple original 3D race scene with Blender when available."""
    blender = shutil.which("blender")
    if not blender:
        raise RuntimeError("Blender is not installed.")

    work_dir = Path("work") / "blender_render"
    work_dir.mkdir(parents=True, exist_ok=True)
    payload_path = work_dir / "race_payload.json"
    script_path = work_dir / "render_race.py"
    payload_path.write_text(json.dumps(_json_safe_payload(payload), ensure_ascii=False), encoding="utf-8")
    script_path.write_text(_BLENDER_SCRIPT, encoding="utf-8")

    command = [
        blender,
        "-b",
        "--python",
        str(script_path),
        "--",
        str(payload_path),
        str(output_path),
    ]
    subprocess.run(command, check=True, timeout=max(180, int(payload.get("duration_sec", 60)) * 8))
    if not Path(output_path).exists():
        raise RuntimeError("Blender rendering finished without creating an MP4 file.")
    return output_path


def _json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    safe.pop("fonts", None)
    return safe


_BLENDER_SCRIPT = r'''
import json
import math
import sys
from pathlib import Path

import bpy


payload_path = Path(sys.argv[-2])
output_path = Path(sys.argv[-1])
payload = json.loads(payload_path.read_text(encoding="utf-8"))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene = bpy.context.scene
scene.render.resolution_x = int(payload["width"])
scene.render.resolution_y = int(payload["height"])
scene.render.fps = int(payload["fps"])
scene.frame_start = 1
scene.frame_end = len(payload["frames"])
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.filepath = str(output_path)


def mat(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


surface = payload["race_config"].get("surface", "芝")
track_mat = mat("track", (0.23, 0.55, 0.28, 1.0) if surface == "芝" else (0.62, 0.42, 0.23, 1.0))
infield_mat = mat("infield", (0.10, 0.28, 0.14, 1.0))
rail_mat = mat("rail", (0.95, 0.94, 0.88, 1.0))

FRAME_COLORS = {
    1: "#FFFFFF",
    2: "#000000",
    3: "#FF0000",
    4: "#0000FF",
    5: "#FFFF00",
    6: "#008000",
    7: "#FFA500",
    8: "#FFC0CB",
}


def rgba_from_hex(value):
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) / 255.0 for index in (0, 2, 4)) + (1.0,)


def text_color_for_frame(frame):
    return (0.02, 0.02, 0.02, 1.0) if frame in [1, 5, 8] else (1.0, 1.0, 1.0, 1.0)


def create_horse_marker(horse_number, frame, location, radius=0.35):
    number = str(horse_number)
    marker_mat = mat(f"frame_{frame}_{number}", rgba_from_hex(FRAME_COLORS.get(frame, "#999999")))
    border_mat = mat(f"marker_border_{number}", (1.0, 1.0, 1.0, 1.0) if frame == 2 else (0.02, 0.02, 0.02, 1.0))
    text_mat = mat(f"marker_text_{number}", text_color_for_frame(frame))

    x, y, z = location
    bpy.ops.mesh.primitive_cylinder_add(vertices=72, radius=radius, depth=0.045, location=(x, y, z))
    border = bpy.context.object
    border.name = f"Marker border {number}"
    border.data.materials.append(border_mat)

    bpy.ops.mesh.primitive_cylinder_add(vertices=72, radius=radius * 0.85, depth=0.065, location=(x, y, z + 0.04))
    marker = bpy.context.object
    marker.name = f"Marker {number}"
    marker.data.materials.append(marker_mat)

    bpy.ops.object.text_add(location=(x, y, z + 0.10), rotation=(0, 0, 0))
    txt = bpy.context.object
    txt.name = f"Label {number}"
    txt.data.body = number
    txt.data.align_x = "CENTER"
    txt.data.align_y = "CENTER"
    txt.data.size = radius * 0.92
    txt.data.materials.append(text_mat)
    return border, marker, txt

bpy.ops.mesh.primitive_torus_add(major_radius=5.0, minor_radius=0.55, location=(0, 0, 0))
track = bpy.context.object
track.name = "Oval course"
track.scale.y = 0.62
track.data.materials.append(track_mat)

bpy.ops.mesh.primitive_torus_add(major_radius=4.25, minor_radius=0.035, location=(0, 0, 0.28))
inner = bpy.context.object
inner.name = "Inner rail"
inner.scale.y = 0.62
inner.data.materials.append(rail_mat)

bpy.ops.mesh.primitive_torus_add(major_radius=5.75, minor_radius=0.035, location=(0, 0, 0.28))
outer = bpy.context.object
outer.name = "Outer rail"
outer.scale.y = 0.62
outer.data.materials.append(rail_mat)

bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=3.8, depth=0.05, location=(0, 0, -0.04))
infield = bpy.context.object
infield.name = "Infield"
infield.scale.y = 0.62
infield.data.materials.append(infield_mat)

marker_objects = {}
text_objects = {}
first_frame = payload["frames"][0]["horses"]
for horse in first_frame:
    number = str(horse["horse_number"])
    frame = int(horse.get("frame", 1))
    border, marker, txt = create_horse_marker(number, frame, (0, 0, 0.39), radius=0.34)
    marker_objects[number] = (border, marker)
    text_objects[number] = txt

for index, frame in enumerate(payload["frames"], start=1):
    scene.frame_set(index)
    for horse in frame["horses"]:
        number = str(horse["horse_number"])
        x = (float(horse["x"]) / payload["width"] - 0.5) * 10.0
        y = (float(horse["y"]) / payload["height"] - 0.5) * -6.2
        border, marker = marker_objects[number]
        border.location = (x, y, 0.39)
        marker.location = (x, y, 0.43)
        border.keyframe_insert(data_path="location")
        marker.keyframe_insert(data_path="location")
        txt = text_objects[number]
        txt.location = (x, y, 0.49)
        txt.keyframe_insert(data_path="location")

bpy.ops.object.light_add(type="AREA", location=(0, -4, 7))
light = bpy.context.object
light.name = "Broadcast softbox"
light.data.energy = 650
light.data.size = 6

bpy.ops.object.camera_add(location=(0, -8.5, 5.5), rotation=(math.radians(58), 0, 0))
scene.camera = bpy.context.object

bpy.ops.wm.save_as_mainfile(filepath=str(payload_path.with_suffix(".blend")))
bpy.ops.render.render(animation=True)
'''
