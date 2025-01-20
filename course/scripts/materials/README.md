# course : scripts : materials
Generate BOM and packing lists

### Materials Format
Name, Level, Description, Quantity, Datasheet (optional), Supplier (optional), Package, x(mm), y(mm), z(mm)

### Bill of Materials (BOM) Format (per course)
Name, Description, Quantity, Datasheet (optional), Link (optional)

### Packing List Format (per course)
Name, Description, Quantity, Package, #Kits, #Required, #Available, #Order, #Ordered, Supplier (optional), x(mm), y(mm), z(mm)

### Package Names
Loose (packed in original packaging, dimensions reflect this package size)
Hardware (screws, standoffs, etc.)
Passive (non-static sensitive components)
Active (static sensitive components)
Audio (mic and speaker PCBs - custom box)
Mounts (laser cut acrylic pieces)
Hindbrain (Custom FPGA package)
Magnets (neodymium magnet case)

## CAD generation for packaging
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