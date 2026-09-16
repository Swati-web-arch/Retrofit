import openstudio

# Create model
model = openstudio.model.Model()

# Building
building = model.getBuilding()
building.setName("RetrofitIQ_Test_Office")

# Create two simple rectangular spaces
for floor in range(2):

    space = openstudio.model.Space(model)
    space.setName(f"Office_Floor_{floor + 1}")

    # 25 m x 20 m rectangle
    width = 25.0
    depth = 20.0
    height = 3.0
    z = floor * height

    # Bottom vertices
    p1 = openstudio.Point3d(0, 0, z)
    p2 = openstudio.Point3d(width, 0, z)
    p3 = openstudio.Point3d(width, depth, z)
    p4 = openstudio.Point3d(0, depth, z)

    # Top vertices
    p5 = openstudio.Point3d(0, 0, z + height)
    p6 = openstudio.Point3d(width, 0, z + height)
    p7 = openstudio.Point3d(width, depth, z + height)
    p8 = openstudio.Point3d(0, depth, z + height)

    # Floor
    floor_surface = openstudio.model.Surface(
        openstudio.Point3dVector([p1, p2, p3, p4]),
        model
    )
    floor_surface.setName(f"Floor_{floor + 1}")
    floor_surface.setSurfaceType("Floor")
    floor_surface.setSpace(space)

    # Ceiling
    ceiling_surface = openstudio.model.Surface(
        openstudio.Point3dVector([p5, p8, p7, p6]),
        model
    )
    ceiling_surface.setName(f"Ceiling_{floor + 1}")
    ceiling_surface.setSurfaceType("RoofCeiling")
    ceiling_surface.setSpace(space)

    # Walls
    walls = [
        [p1, p5, p6, p2],
        [p2, p6, p7, p3],
        [p3, p7, p8, p4],
        [p4, p8, p5, p1],
    ]

    for i, wall_points in enumerate(walls):
        wall = openstudio.model.Surface(
            openstudio.Point3dVector(wall_points),
            model
        )
        wall.setName(f"Floor_{floor + 1}_Wall_{i + 1}")
        wall.setSurfaceType("Wall")
        wall.setSpace(space)

# Save
output_path = (
    r"C:\Users\Lenovo\sttttttorage\Documents"
    r"\Retrofit\simulation\RetrofitIQ_test.osm"
)

model.save(output_path, True)

print("Model created successfully!")
print(output_path)