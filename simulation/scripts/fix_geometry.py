from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSM_FILE = PROJECT_ROOT / "simulation" / "RetrofitIQ_baseline_hvac_fixed.osm"

text = OSM_FILE.read_text(encoding="utf-8")


def fix_surface(text, surface_name, boundary, boundary_object,
                sun_exposure, wind_exposure, vertices):
    """
    Modify one OS:Surface object by its name.
    """

    pattern = re.compile(
        r"(OS:Surface,\s*"
        r".*?\n\s*" + re.escape(surface_name) + r",.*?"
        r"\n\s*[^;]+;)",
        re.DOTALL
    )

    match = pattern.search(text)

    if not match:
        raise ValueError(f"Could not find surface: {surface_name}")

    block = match.group(1)

    lines = block.splitlines()

    # Find the fields using their comments.
    for i, line in enumerate(lines):
        if "!- Outside Boundary Condition" in line:
            lines[i] = f"    {boundary}, !- Outside Boundary Condition"

        elif "!- Outside Boundary Condition Object" in line:
            if boundary_object:
                lines[i] = (
                    f"    {boundary_object}, "
                    f"!- Outside Boundary Condition Object"
                )
            else:
                lines[i] = (
                    "    , !- Outside Boundary Condition Object"
                )

        elif "!- Sun Exposure" in line:
            lines[i] = f"    {sun_exposure}, !- Sun Exposure"

        elif "!- Wind Exposure" in line:
            lines[i] = f"    {wind_exposure}, !- Wind Exposure"

    # Replace the vertex section.
    vertex_start = None

    for i, line in enumerate(lines):
        if "!- Number of Vertices" in line:
            vertex_start = i + 1
            break

    if vertex_start is None:
        raise ValueError(
            f"Could not find vertex section for {surface_name}"
        )

    # Keep everything before vertices.
    prefix = lines[:vertex_start]

    vertex_lines = [
        f"    {len(vertices)}, !- Number of Vertices"
    ]

    for i, (x, y, z) in enumerate(vertices, start=1):
        ending = ";" if i == len(vertices) else ","
        vertex_lines.append(
            f"    {x}, {y}, {z}{ending} !- X,Y,Z Vertex {i} {{m}}"
        )

    # The old vertex count line is already in prefix, so remove
    # everything after it.
    new_block_lines = prefix[:-1] + vertex_lines

    new_block = "\n".join(new_block_lines)

    return text[:match.start(1)] + new_block + text[match.end(1):]


# ============================================================
# CORRECT GEOMETRY
# ============================================================

# Floor 1:
# Ground-facing floor -> normal points downward.
text = fix_surface(
    text,
    "Floor_1",
    "Ground",
    "",
    "NoSun",
    "NoWind",
    [
        (0, 20, 0),
        (25, 20, 0),
        (25, 0, 0),
        (0, 0, 0),
    ],
)

# Ceiling 1:
# Interior ceiling paired with Floor 2.
text = fix_surface(
    text,
    "Ceiling_1",
    "Surface",
    "Floor_2",
    "NoSun",
    "NoWind",
    [
        (0, 0, 3),
        (25, 0, 3),
        (25, 20, 3),
        (0, 20, 3),
    ],
)

# Floor 2:
# Interior floor paired with Ceiling 1.
text = fix_surface(
    text,
    "Floor_2",
    "Surface",
    "Ceiling_1",
    "NoSun",
    "NoWind",
    [
        (0, 20, 3),
        (25, 20, 3),
        (25, 0, 3),
        (0, 0, 3),
    ],
)

# Ceiling 2:
# Top roof -> outdoors.
text = fix_surface(
    text,
    "Ceiling_2",
    "Outdoors",
    "",
    "SunExposed",
    "WindExposed",
    [
        (0, 0, 6),
        (25, 0, 6),
        (25, 20, 6),
        (0, 20, 6),
    ],
)


OSM_FILE.write_text(text, encoding="utf-8")

print("Geometry correction applied successfully.")
print(f"Updated: {OSM_FILE}")