#!/usr/bin/env python3
"""
Generate an interactive 3D point cloud viewer (HTML) from a sparse reconstruction directory.

Usage:
    python view_pointcloud.py <dir_path> [--max-points N]

Examples:
    python view_pointcloud.py /app/vggt/2b6b4f0a1f6499c6d3f196e2f48fd09c
    python view_pointcloud.py /app/vggt/719bccb114a55c9e76dabd87f3a8ab65 --max-points 50000

The script looks for sparse/points.ply inside the given directory and outputs
pointcloud_viewer.html in the same directory.
"""

import argparse
import base64
import os
import sys

import numpy as np
import trimesh


def find_ply(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "sparse", "points.ply"),
        os.path.join(base_dir, "sparse", "gaussians.ply"),
        os.path.join(base_dir, "points.ply"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # If the path itself is a .ply file
    if base_dir.endswith(".ply") and os.path.isfile(base_dir):
        return base_dir
    raise FileNotFoundError(
        f"No PLY file found. Searched:\n  " + "\n  ".join(candidates)
    )


def load_points(ply_path: str, max_points: int):
    mesh = trimesh.load(ply_path)
    verts = np.array(mesh.vertices, dtype=np.float32)

    if hasattr(mesh, "colors") and mesh.colors is not None and len(mesh.colors) > 0:
        colors = np.array(mesh.colors, dtype=np.uint8)[:, :3]
    elif hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        colors = np.array(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3]
    else:
        colors = np.full((len(verts), 3), 180, dtype=np.uint8)

    if len(verts) > max_points:
        idx = np.random.RandomState(42).choice(len(verts), size=max_points, replace=False)
        verts = verts[idx]
        colors = colors[idx]

    center = verts.mean(axis=0)
    verts -= center
    scale = np.abs(verts).max()
    if scale > 0:
        verts /= scale

    return verts, colors


def generate_html(verts: np.ndarray, colors: np.ndarray, title: str) -> str:
    n = len(verts)
    pos_b64 = base64.b64encode(verts.astype(np.float32).tobytes()).decode()
    col_b64 = base64.b64encode(colors.astype(np.uint8).tobytes()).decode()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a0f; overflow: hidden; font-family: 'JetBrains Mono', monospace; }}
  canvas {{ display: block; }}
  #info {{
    position: absolute; top: 16px; left: 16px;
    color: #8899aa; font-size: 13px; line-height: 1.6;
    background: rgba(10,10,15,0.85); padding: 12px 16px;
    border: 1px solid #223; border-radius: 6px;
    pointer-events: none;
  }}
  #info span {{ color: #aaddff; }}
  #controls {{
    position: absolute; bottom: 16px; left: 16px;
    color: #8899aa; font-size: 12px;
    background: rgba(10,10,15,0.85); padding: 10px 14px;
    border: 1px solid #223; border-radius: 6px;
  }}
  #controls label {{ display: block; margin: 4px 0; cursor: pointer; }}
  #controls input[type=range] {{ width: 120px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="info">
  <b>{title}</b><br>
  Points: <span>{n:,}</span><br>
  Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan
</div>
<div id="controls">
  <label>Point size: <input type="range" id="sizeSlider" min="1" max="30" value="6" step="1">
    <span id="sizeVal">0.006</span>
  </label>
</div>
<script type="importmap">
{{ "imports": {{ "three": "https://esm.sh/three@0.162.0", "three/addons/": "https://esm.sh/three@0.162.0/examples/jsm/" }} }}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 100);
camera.position.set(0, 0, 2.5);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.8;

const N = {n};

function b64ToArray(b64, Type) {{
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}}

const positions = b64ToArray("{pos_b64}", Float32Array);
const colorsRaw = b64ToArray("{col_b64}", Uint8Array);
const colorsFloat = new Float32Array(N * 3);
for (let i = 0; i < N * 3; i++) colorsFloat[i] = colorsRaw[i] / 255.0;

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colorsFloat, 3));

const material = new THREE.PointsMaterial({{
  size: 0.006,
  vertexColors: true,
  sizeAttenuation: true,
  transparent: true,
  opacity: 0.9,
}});

const points = new THREE.Points(geometry, material);
scene.add(points);

const axes = new THREE.AxesHelper(0.3);
axes.material.opacity = 0.3;
axes.material.transparent = true;
scene.add(axes);

const sizeSlider = document.getElementById('sizeSlider');
const sizeVal = document.getElementById('sizeVal');
sizeSlider.addEventListener('input', () => {{
  const s = sizeSlider.value / 1000;
  material.size = s;
  sizeVal.textContent = s.toFixed(3);
}});

addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

(function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate 3D point cloud viewer HTML")
    parser.add_argument("path", help="Directory containing sparse/ subfolder, or path to a .ply file")
    parser.add_argument("--max-points", type=int, default=100_000, help="Max points to display (default: 100000)")
    parser.add_argument("-o", "--output", help="Output HTML path (default: <dir>/pointcloud_viewer.html)")
    args = parser.parse_args()

    ply_path = find_ply(args.path)
    print(f"Loading: {ply_path}")

    verts, colors = load_points(ply_path, args.max_points)
    print(f"Points: {len(verts):,}")

    title = os.path.basename(os.path.dirname(ply_path) if ply_path.endswith(".ply") else args.path)
    if title == "sparse":
        title = os.path.basename(os.path.dirname(os.path.dirname(ply_path)))

    out_dir = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    out_path = args.output or os.path.join(out_dir, "pointcloud_viewer.html")

    html = generate_html(verts, colors, title)
    with open(out_path, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Written: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
