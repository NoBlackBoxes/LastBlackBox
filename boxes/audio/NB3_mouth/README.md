# Audio : NB3_mouth

PCB design and fabrication for NB3 mouth, a mono i2s DAC and Amplifier

## Parts

- MAX98357A from Maxim: [datasheet](libraries/parts/I2S_DAC-AMP_MAX98357A/I2S_DAC-AMP_MAX98357A.pdf)

## Fabrication

### Generate files
0. Save front/back renders and 3D model (step)
1. Create "fab" folder in KiCAD project root
2. Export BOM (as CSV): File->Fabrication outputs->BOM
3. Edit BOM: Open as ODS and save as XLSX in "fab" folder
4. Generate Gerbers: File->Fabrication outputs->Gerbers [settings](NB3_mouth_FAB_plot_settings.png)
5. Generate Drill: File->Fabrication outputs->Drill Files [settings](NB3_mouth_FAB_drill_settings.png)
6. Generate Centroids: File->Fabrication outputs->Componenet Placement [settings](NB3_mouth_FAB_pos_settings.png)
7. Zip entire fab folder (including BOM.xlsx)

### Upload boards(s) - PCBWay

#### Assembly Service
- Service: Turnkey Assembly
- Board type: Panelized PCBs
- Assembly: Both Sides
- Quantity: *80*

##### Other Parameters
- Number of unique Parts: 6
- Number of SMD Parts: 6
- Number of BGA/QFP Parts: 1
- Number of Through-Hole Parts: 1
			
#### PCB Specifications
- Board type: Panel by PCBWay
- Panel requirements: Panel in 2*4, total 20 panels=total 160 individual boards.
- Route Process: Panel as Tab Route
- X-out Allowance in Panel: Accept
- Different design in panel: 1
- Size: 56 x 78 mm
- Quantity: 20
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
