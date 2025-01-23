# Systems : NB3_eye

PCB design and fabrication for NB3 eye, an adapter board to install 8x8 LED matrices on the NB3

## Parts

- Just connectors

## Fabrication

### Generate files
0. Save front/back renders and 3D model (step) (3D models of each board half, left and right eye are more useful)
1. Create "fab" folder in KiCAD project root
2. Export BOM (as CSV): File->Fabrication outputs->BOM
3. Edit BOM: Open as ODS and save as XLSX in "fab" folder
4. Generate Gerbers: File->Fabrication outputs->Gerbers [settings](settings/NB3_eye_FAB_plot_settings.png)
5. Generate Drill: File->Fabrication outputs->Drill Files [settings](settings/NB3_eye_FAB_drill_settings.png)
6. Generate Centroids: File->Fabrication outputs->Component Placement [settings](settings/NB3_eye_FAB_pos_settings.png)
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
- Number of Through-Hole Parts: 6
			
#### PCB Specifications
- Board type: Panel by PCBWay
- Panel Requirements: 
    - Break away rail: No
    - Please use V-scoring for the panel. The board files contain two distinct designs seperated by a V-score line indicated in the User.Eco1 layer.
    - Route process: Panel as V-scoring
- Different design in panel: 2
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
