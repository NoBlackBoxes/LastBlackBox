# course : tools : packaging
Generate packaging for LBB course kits

## CAD generation for course material models
We use the open-source CadQuery Python library, which uses the OpenCascade kernel

```bash
pip install git+https://github.com/CadQuery/cadquery.git
# only the most recent version in compatible with Python 3.13
```

```python
# CadQuery Example
import cadquery as cq

# Parameters for the cuboid
length = 50  # Length of the cuboid
width = 30   # Width of the cuboid
height = 20  # Height of the cuboid
output_file = "cuboid.step"  # Output STEP file name

# Create the cuboid
cuboid = cq.Workplane("XY").box(length, width, height)

# Export the cuboid to a STEP file
cq.exporters.export(cuboid, output_file)

print(f"STEP file '{output_file}' has been created.")
```