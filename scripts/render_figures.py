"""Render SVG notebook figures using the system librsvg and Cairo libraries."""

import argparse
import ctypes as c
import hashlib
import json
import subprocess
from pathlib import Path
import re


class Rectangle(c.Structure):
    _fields_ = [
        ("x", c.c_double),
        ("y", c.c_double),
        ("width", c.c_double),
        ("height", c.c_double),
    ]


def render_all(source, output):
    svg = c.CDLL("librsvg-2.so.2")
    cairo = c.CDLL("libcairo.so.2")
    gobj = c.CDLL("libgobject-2.0.so.0")
    svg.rsvg_handle_new_from_file.argtypes = [c.c_char_p, c.c_void_p]
    svg.rsvg_handle_new_from_file.restype = c.c_void_p
    svg.rsvg_handle_render_document.argtypes = [
        c.c_void_p,
        c.c_void_p,
        c.POINTER(Rectangle),
        c.c_void_p,
    ]
    svg.rsvg_handle_render_document.restype = c.c_int
    cairo.cairo_image_surface_create.argtypes = [c.c_int, c.c_int, c.c_int]
    cairo.cairo_image_surface_create.restype = c.c_void_p
    cairo.cairo_create.argtypes = [c.c_void_p]
    cairo.cairo_create.restype = c.c_void_p
    cairo.cairo_surface_write_to_png.argtypes = [c.c_void_p, c.c_char_p]
    cairo.cairo_surface_write_to_png.restype = c.c_int
    cairo.cairo_destroy.argtypes = [c.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [c.c_void_p]
    gobj.g_object_unref.argtypes = [c.c_void_p]
    output.mkdir(parents=True, exist_ok=True)
    entries = {}
    for path in sorted(source.glob("*.svg")):
        content = path.read_text()
        width = int(re.search(r'width="(\d+)"', content).group(1))
        height = int(re.search(r'height="(\d+)"', content).group(1))
        handle = svg.rsvg_handle_new_from_file(str(path).encode(), None)
        if not handle:
            raise ValueError(f"cannot read SVG {path}")
        surface = cairo.cairo_image_surface_create(0, width, height)
        context = cairo.cairo_create(surface)
        try:
            viewport = Rectangle(0, 0, width, height)
            if not svg.rsvg_handle_render_document(
                handle, context, c.byref(viewport), None
            ):
                raise ValueError(f"cannot render SVG {path}")
            if cairo.cairo_surface_write_to_png(
                surface, str(output / f"{path.stem}.png").encode()
            ):
                raise ValueError(f"cannot write PNG for {path}")
        finally:
            cairo.cairo_destroy(context)
            cairo.cairo_surface_destroy(surface)
            gobj.g_object_unref(handle)
        preview = output / f"{path.stem}.png"
        entries[path.name] = {
            "svg_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "png_sha256": hashlib.sha256(preview.read_bytes()).hexdigest(),
        }
    font_result = subprocess.run(
        ["fc-match", "sans", "-f", "%{file}"],
        capture_output=True,
        text=True,
        check=True,
    )
    font = Path(font_result.stdout)
    try:
        version = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", "librsvg2-2:amd64"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        version = "unavailable"
    cairo.cairo_version_string.restype = c.c_char_p
    manifest = {
        "renderer": "librsvg + Cairo",
        "librsvg_version": version,
        "cairo_version": cairo.cairo_version_string().decode(),
        "font": font.name,
        "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
        "figures": entries,
    }
    (output / "preview-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("notebooks/assets/architecture")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("notebooks/assets/architecture/previews")
    )
    args = parser.parse_args()
    render_all(args.source, args.output)


if __name__ == "__main__":
    main()
