# The Last Black Box: Course: Scripts: Renders
Instructions and helper scripts for creating renders of the NB3 and its components

## Export for KiCAD
For rendering purposes, export a VRML file.
 - Set the board colors in "board setup" (solder mask and silkscreen)
 - Export models to a folder called "3D" within the project folder
 - Use the "board center"
 - Units: mm
 - Copy 3D files
 - Use relative paths

 The exported WRL file and folder with required 3D models can be imported into Blender (after installing the Web 3D/VRML2 add on) and FreeCAD. They include the mesh geometry and textures.

## Importing VRML into Blender
- Set the Units to mm
- Import VRRML2
  - Change the scale to 1.0000 (might help to downsize to mm)
  - Change the forward and up vectors (if required)
  - Select import "as collection"
- Many objects will appear in the import collection, select them all and "Join" into a single object
- Change the scale to .00254

## Importing DXF into FreeCAD
- Import DXF (make sure the scaling is correct)
- Select all the elements of the drawing
- From "Draft" workbench, select "Draft to Sketch"
- Create a new (empty) body
- Add sketch to new Body, then pad, etc.

## Hints, Tips, Tricks, etc.
- VRML looks good in blender, but is not vey configurable
- VRML in FreeCAD looks good, but is a pain t position
- You can arrange assemblies in FreeCAD (eveything except VRML assests) and export them as GLTF, which can then be imported into Blender for rendering

## Movies

```bash
# Assemble movie from PNGs
ffmpeg -r 30 -f image2 -pattern_type glob -i '*.png' -s 1920x1080 -vcodec libx264 -pix_fmt yuv420p
 animation.mp4
# -pix_fmt yuv420p necessary for WhatsApp


# Loop movie (assumes 4 loops and 2x240 frames, 30 FPS)
ffmpeg -i animation.mp4 -filter_complex "[0]reverse[r];[0][r]concat,loop=4:480,setpts=N/30/TB" loop.mp4
```