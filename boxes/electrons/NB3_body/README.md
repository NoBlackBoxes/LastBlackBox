# Electrons : NB3_body

PCB design and fabrication for NB3 body, a brain-shaped electronics prototyping base

## Parts

- Just connnectors and passives

## Fabrication

### Generate files
0. Save front/back renders and 3D model (step)
1. Create "fab" folder in KiCAD project root
2. Export BOM (as CSV): File->Fabrication outputs->BOM
3. Edit BOM: Open as ODS and save as XLSX in "fab" folder
4. Generate Gerbers: File->Fabrication outputs->Gerbers [settings](NB3_body_FAB_plot_settings.png)
5. Generate Drill: File->Fabrication outputs->Drill Files [settings](NB3_body_FAB_drill_settings.png)
6. Generate Centroids: File->Fabrication outputs->Componenet Placement [settings](NB3_body_FAB_pos_settings.png)
7. Zip entire fab folder (including BOM.xlsx)

### Upload boards(s) - PCBWay

#### Assembly Service
- Service: Turnkey Assembly
- Board type: Single Pieces
- Assembly: Top-Side
- Quantity: *50*

##### Other Parameters
- Number of unique Parts: 4
- Number of SMD Parts: 14
- Number of BGA/QFP Parts: 0
- Number of Through-Hole Parts: 0
			
#### PCB Specifications
- Board type: Single Pieces
- Size: 200 x 200 mm
- Quantity: 50
- Layer: 2 Layers
- Material: FR-4: TG150-160
- Thickness: 1.6 mm
- Min track/spacing: 6/6mil
- Min hole size: 0.3mm
- Solder mask: White
- Silkscreen: Black
- Edge connector: No
- Surface finish: Immersion gold(ENIG) (1U"), "HASL" to "ENIG" No
- Via process: Tenting vias
- Finished copper: 1 oz Cu
- Remove product No.: No
