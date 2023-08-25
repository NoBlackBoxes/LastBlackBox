# FPGAs : NB3_hindbrain

PCB design and fabrication for NB3 hindbrain, a FPGA development board

## Parts

Name|Description|Mfg|Mfg#|Sup|Sup#|Link|
:--:|:---------:|:-:|:--:|:-:|:--:|:--:|
Micro-USB|Micro yype-B USB 2.0 SMD socket|Amphenol|10103594-0001LF|LCSC|C428495|[datasheet](NB3_hindbrain/libraries/parts/MicroUSB_10103594-0001LF/MicroUSB_10103594-0001LF.pdf)
EEPROM|2K microwire compatible serial EEPROM|Microchip|93LC56BT-I/OT|LCSC|C190271|[datasheet](NB3_hindbrain/libraries/parts/EEPROM_93LC56BT-I-OT/EEPROM_93LC56BT-I-OT.pdf)
USB-to-Serial|USB HS interface IC to dual UART/FIFO/SPI/JTAG/I2C|FTDI|FT2232HQ|Mouser|895-FT2232HQ|[datasheet](NB3_hindbrain/libraries/parts/USBtoSerial_FT2232HQ/USBtoSerial_FT2232HQ.pdf)
Ferrite Bead|GHz noise suppression SMD ferrite bead|Murata|BLM18HE152SN1D|LCSC|C82155|[datasheet](NB3_hindbrain/libraries/parts/FB_BLM18HE152SN1D/FB_BLM18HE152SN1D.pdf)
LDO 1.2V|Low dropout voltage regulator 800mA 1.2V|STM|LD1117S12TR|LCSC|C155612|[datasheet](NB3_hindbrain/libraries/parts/LDO_1v2_LD1117S12TR/LDO_1v2_LD1117S12TR.pdf)
LDO 3.3V|Low dropout voltage regulator 800mA 3.3V|Texas Instruments|LM1117IMPX-3.3|LCSC|C23984|[datasheet](NB3_hindbrain/libraries/parts/LDO_3v3_LM1117IMPX-3.3/LDO_3v3_LM1117IMPX-3.3.pdf)
RGB LED|Surface mount high-intensity RGB LED|Cree|CLMVC-FKA-CLBDGL7LBB79353|Mouser|941-CLMVCFKACLBDGL7L|[datasheet](NB3_hindbrain/libraries/parts/RGB_CLMVC-FKA-CLBDGL7LBB79353/RGB_CLMVC-FKA-CLBDGL7LBB79353.pdf)
FPGA|Lattice ICE40 UltraPlus 5280 LUTs 1.2V|Lattice|ICE40UP5K-SG48I|Mouser|842-ICE40UP5K-SG48I|[datasheet](NB3_hindbrain/libraries/parts/FPGA_ICE40UP5K-SG48I/FPGA_ICE40UP5K-SG48I.pdf)
NOR Flash|32M-bit 4Kb uniform sector SPI flash|WinBond|W25Q32JVSSIQ|LCSC|C179173|[datasheet](NB3_hindbrain/libraries/parts/NOR_Flash_W25Q32JVSSIQ/NOR_Flash_W25Q32JVSSIQ.pdf)
Oscillator|Clock oscillators 12MHz 1.6-3.6V|ECS|ECS-2520MV-120-BN-TR|Mouser|520-2520MV-120-BN-T|[datasheet](NB3_hindbrain/libraries/parts/Osc_12MHz_ECS-2520MV-120-BN-TR/Osc_12MHz_ECS-2520MV-120-BN-TR.pdf)
ADC|10-bit SPI 4 channel ADC|Microchip|MCP3004-I/SL|Mouser|579-MCP3004-I/SL|[datasheet](NB3_hindbrain/libraries/parts/ADC_MCP3004-I-SL/ADC_MCP3004-I-SL.pdf)
Pin Header|1x20 2.54 mm through-hole pin header|BOOMELE|C50981|LCSC|C50981|[datasheet](NB3_hindbrain/libraries/parts/PinHeader_C50981/PinHeader_C50981.pdf)

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
