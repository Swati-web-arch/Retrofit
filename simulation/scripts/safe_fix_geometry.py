from pathlib import Path
import re
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSM_FILE = PROJECT_ROOT / "simulation" / "RetrofitIQ_baseline_hvac_fixed.osm"
BACKUP_FILE = PROJECT_ROOT / "simulation" / "RetrofitIQ_baseline_hvac_fixed_before_safe_geometry.osm"


def replace_field(block, field_name, value):
    pattern = rf"(?m)^(\s*).*?!-\s*{re.escape(field_name)}\s*$"

    def repl(match):
        indent = match.group(1)
        comment = block[match.start():match.end()].split("!-", 1)[1]
        return f"{indent}{value}, !- {comment}"

    new_block, count = re.subn(pattern, repl, block)

    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 '{field_name}' field, found {count}"
        )

    return new_block


def replace_vertex(block, vertex_number, coordinates, final_vertex=False):
    x, y, z = coordinates

    pattern = (
        rf"(?m)^(\s*).*?!-\s*X,Y,Z Vertex "
        rf"{vertex_number} \{{m\}}[^\r\n]*$"
    )

    def repl(match):
        indent = match.group(1)
        ending = ";" if final_vertex else ","
        return (
            f"{indent}{x}, {y}, {z}{ending}"
            f" !- X,Y,Z Vertex {vertex_number} {{m}}"
        )

    new_block, count = re.subn(pattern, repl, block)

    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 vertex {vertex_number}, found {count}"
        )

    return new_block


def get_surface_blocks(text):
    pattern = r"(?ms)^OS:Surface,\r?\n.*?(?=\r?\n\r?\n|\Z)"
    return list(re.finditer(pattern, text))


def get_surface_name(block):
    match = re.search(
        r"(?m)^\s*([^,\r\n]+),\s*!-\s*Name\s*$",
        block
    )

    if not match:
        raise RuntimeError("Could not find OS:Surface name")

    return match.group(1).strip()


def fix_surface(block, name):
    if name == "Floor_1":
        block = replace_field(
            block,
            "Outside Boundary Condition",
            "Ground"
        )
        block = replace_field(
            block,
            "Outside Boundary Condition Object",
            ""
        )
        block = replace_field(
            block,
            "Sun Exposure",
            "NoSun"
        )
        block = replace_field(
            block,
            "Wind Exposure",
            "NoWind"
        )

        block = replace_vertex(block, 1, (0, 20, 0))
        block = replace_vertex(block, 2, (25, 20, 0))
        block = replace_vertex(block, 3, (25, 0, 0))
        block = replace_vertex(block, 4, (0, 0, 0), final_vertex=True)

    elif name == "Ceiling_1":
        block = replace_field(
            block,
            "Outside Boundary Condition",
            "Surface"
        )
        block = replace_field(
            block,
            "Outside Boundary Condition Object",
            "Floor_2"
        )
        block = replace_field(
            block,
            "Sun Exposure",
            "NoSun"
        )
        block = replace_field(
            block,
            "Wind Exposure",
            "NoWind"
        )

        block = replace_vertex(block, 1, (0, 0, 3))
        block = replace_vertex(block, 2, (25, 0, 3))
        block = replace_vertex(block, 3, (25, 20, 3))
        block = replace_vertex(block, 4, (0, 20, 3), final_vertex=True)

    elif name == "Floor_2":
        block = replace_field(
            block,
            "Outside Boundary Condition",
            "Surface"
        )
        block = replace_field(
            block,
            "Outside Boundary Condition Object",
            "Ceiling_1"
        )
        block = replace_field(
            block,
            "Sun Exposure",
            "NoSun"
        )
        block = replace_field(
            block,
            "Wind Exposure",
            "NoWind"
        )

        block = replace_vertex(block, 1, (0, 20, 3))
        block = replace_vertex(block, 2, (25, 20, 3))
        block = replace_vertex(block, 3, (25, 0, 3))
        block = replace_vertex(block, 4, (0, 0, 3), final_vertex=True)

    elif name == "Ceiling_2":
        block = replace_field(
            block,
            "Outside Boundary Condition",
            "Outdoors"
        )
        block = replace_field(
            block,
            "Outside Boundary Condition Object",
            ""
        )
        block = replace_field(
            block,
            "Sun Exposure",
            "SunExposed"
        )
        block = replace_field(
            block,
            "Wind Exposure",
            "WindExposed"
        )

        block = replace_vertex(block, 1, (0, 0, 6))
        block = replace_vertex(block, 2, (25, 0, 6))
        block = replace_vertex(block, 3, (25, 20, 6))
        block = replace_vertex(block, 4, (0, 20, 6), final_vertex=True)

    return block


print(f"OSM file: {OSM_FILE}")

if not OSM_FILE.exists():
    raise FileNotFoundError(f"OSM file not found: {OSM_FILE}")

# Safety backup
shutil.copy2(OSM_FILE, BACKUP_FILE)
print(f"Backup created: {BACKUP_FILE}")

text = OSM_FILE.read_text(encoding="utf-8")

surface_blocks = get_surface_blocks(text)

targets = {
    "Floor_1",
    "Ceiling_1",
    "Floor_2",
    "Ceiling_2",
}

found = set()

for match in reversed(surface_blocks):
    block = match.group(0)
    name = get_surface_name(block)

    if name in targets:
        print(f"Fixing surface: {name}")
        fixed_block = fix_surface(block, name)

        text = (
            text[:match.start()]
            + fixed_block
            + text[match.end():]
        )

        found.add(name)

if found != targets:
    missing = targets - found
    raise RuntimeError(f"Missing target surfaces: {missing}")

OSM_FILE.write_text(text, encoding="utf-8")

print()
print("Geometry correction completed.")
print(f"Updated: {OSM_FILE}")
print("Modified surfaces:", ", ".join(sorted(found)))