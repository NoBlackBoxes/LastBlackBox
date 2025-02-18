# Power : NB3_charger

PCB design and fabrication for NB3 charger, a 5V/3A DC to DC power supply and NiMH battery charger

## Parts
- Battery Holder (Keystone 1028)
  - Substitute LCSC version (C5290180), may require modified footprint
- AP64501SP-13, DC-DC converter, Diodes Inc. [datasheet](libraries/parts/Voltage_Regulator_AP64501/Voltage_Regulator_AP64501.pdf)
- Body Power Connector
  - B4B-PH-SM4-TB (SMD)
  - B4B-PH-K-S (through-hole)

## Fabrication

### Generate files
0. Save front/back renders and 3D model (step)
1. Create "fab" folder in KiCAD project root
2. Export BOM (as CSV): File->Fabrication outputs->BOM
3. Edit BOM: Open as ODS and save as XLSX in "fab" folder
4. Generate Gerbers: File->Fabrication outputs->Gerbers [settings](NB3_charger_FAB_plot_settings.png)
5. Generate Drill: File->Fabrication outputs->Drill Files [settings](NB3_charger_FAB_drill_settings.png)
6. Generate Centroids: File->Fabrication outputs->Component Placement [settings](NB3_charger_FAB_pos_settings.png)
7. Zip entire fab folder (including BOM.xlsx)

### Upload boards(s) - PCBWay

#### Assembly Service
- Service: Turnkey Assembly
- Board type: Single Pieces
- Assembly: Top-Side
- Quantity: *2*

##### Other Parameters
- Number of unique Parts: 28
- Number of SMD Parts: 42
- Number of BGA/QFP Parts: 2
- Number of Through-Hole Parts: 11
			
#### PCB Specifications
- Board type: 
- Panel requirements: 
- Route Process: 
- X-out Allowance in Panel: Accept
- Different design in panel: 1
- Size: 150 x 65 mm
- Quantity: 5
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
