from pathlib import Path
import shutil
import re

PROJECT_ROOT = Path.cwd()

OSM_FILE = PROJECT_ROOT / "simulation" / "RetrofitIQ_baseline_hvac_fixed.osm"
BACKUP_FILE = PROJECT_ROOT / "simulation" / "RetrofitIQ_baseline_hvac_fixed_before_wall_fix.osm"

# Create backup
shutil.copy2(OSM_FILE, BACKUP_FILE)
print(f"Backup created: {BACKUP_FILE}")

# Exact correct vertex order
correct_vertices = {
    "Floor_1_Wall_1": [
        "0, 0, 0,",
        "25, 0, 0,",
        "25, 0, 3,",
        "0, 0, 3;"
    ],

    "Floor_1_Wall_2": [
        "25, 0, 0,",
        "25, 20, 0,",
        "25, 20, 3,",
        "25, 0, 3;"
    ],

    "Floor_1_Wall_3": [
        "25, 20, 0,",
        "0, 20, 0,",
        "0, 20, 3,",
        "25, 20, 3;"
    ],

    "Floor_1_Wall_4": [
        "0, 20, 0,",
        "0, 0, 0,",
        "0, 0, 3,",
        "0, 20, 3;"
    ],

    "Floor_2_Wall_1": [
        "0, 0, 3,",
        "25, 0, 3,",
        "25, 0, 6,",
        "0, 0, 6;"
    ],

    "Floor_2_Wall_2": [
        "25, 0, 3,",
        "25, 20, 3,",
        "25, 20, 6,",
        "25, 0, 6;"
    ],

    "Floor_2_Wall_3": [
        "25, 20, 3,",
        "0, 20, 3,",
        "0, 20, 6,",
        "25, 20, 6;"
    ],

    "Floor_2_Wall_4": [
        "0, 20, 3,",
        "0, 0, 3,",
        "0, 0, 6,",
        "0, 20, 6;"
    ]
}

text = OSM_FILE.read_text(encoding="utf-8")

updated = 0

for name, vertices in correct_vertices.items():

    # Find the complete OS:Surface object having this exact name.
    pattern = (
        r"(OS:Surface,\s*\n"
        r"\s*[^,]+,\s*!-\s*Handle\s*\n"
        r"\s*" + re.escape(name) +
        r",\s*!-\s*Name.*?\n"
        r".*?"
        r"\s*,\s*!-\s*Number of Vertices.*?\n)"
        r"((?:\s*[^!\n]+,\s*!-\s*X,Y,Z Vertex 1.*?\n)"
        r"(?:\s*[^!\n]+,\s*!-\s*X,Y,Z Vertex 2.*?\n)"
        r"(?:\s*[^!\n]+,\s*!-\s*X,Y,Z Vertex 3.*?\n)"
        r"(?:\s*[^!\n]+[;]\s*!-\s*X,Y,Z Vertex 4.*?\n))"
    )

    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        print(f"NOT FOUND: {name}")
        continue

    # Preserve indentation and comments.
    replacement_lines = []

    for i, vertex in enumerate(vertices, start=1):
        ending = ";" if i == 4 else ","
        replacement_lines.append(
            f"  {vertex} !- X,Y,Z Vertex {i} {{m}}\n"
        )

    new_vertex_block = "".join(replacement_lines)

    text = (
        text[:match.start(2)]
        + new_vertex_block
        + text[match.end(2):]
    )

    updated += 1
    print(f"Updated: {name}")

# Save
OSM_FILE.write_text(text, encoding="utf-8")

print()
print(f"Total walls updated: {updated}/8")

if updated == 8:
    print("SUCCESS: All 8 wall surfaces were fixed.")
else:
    print("WARNING: Not all 8 walls were found.")