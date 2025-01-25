# Systems : NB3_eyes

PCB design and fabrication for NB3 "eyes", an adapter board to install 8x8 LED matrices on the NB3

## Parts
- Just connectors

## Fabrication

### Generate files
0. Save front/back renders and 3D model (step and vrml)
1. Create "fab" folder in KiCAD project root
2. Export BOM (as CSV): File->Fabrication outputs->BOM
3. Edit BOM: Open as ODS and save as XLSX in "fab" folder
4. Generate Gerbers: File->Fabrication outputs->Gerbers [settings](settings/NB3_eyes_FAB_plot_settings.png)
5. Generate Drill: File->Fabrication outputs->Drill Files [settings](settings/NB3_eyes_FAB_drill_settings.png)
6. Generate Centroids: File->Fabrication outputs->Component Placement [settings](settings/NB3_eyes_FAB_pos_settings.png)
7. Zip entire fab folder (including BOM.xlsx)

### Upload boards(s) - PCBWay

#### Assembly Service
- Service: Turnkey Assembly
- Board type: Panelized PCBs
- Assembly: Top-Side
- Quantity: *50*

##### Other Parameters
- Number of unique Parts: 3
- Number of SMD Parts: 0
- Number of BGA/QFP Parts: 0
- Number of Through-Hole Parts: 5
			
#### PCB Specifications
- Board type: Panel by PCBWay
- Panel Requirements: 
    - Break away rail: Yes
    - Panel in 1*5, 10 panels=total 50 individual boards.
    - Route process: Panel as V-scoring
- Different design in panel: 1
- Size: 45 x 60 mm
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
