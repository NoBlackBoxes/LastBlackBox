# FPGAs : NB3_hindbrain

PCB design and fabrication for the NB3 hindbrain, an FPGA development board with ADC and DAC

## Parts

Name|Description|Mfg|Mfg#|Sup|Sup#|Link|
:--:|:---------:|:-:|:--:|:-:|:--:|:--:|
Micro-USB|Micro yype-B USB 2.0 SMD socket|Amphenol|10103594-0001LF|LCSC|C428495|[datasheet](libraries/parts/MicroUSB_10103594-0001LF/MicroUSB_10103594-0001LF.pdf)
EEPROM|2K microwire compatible serial EEPROM|Microchip|93LC56BT-I/OT|LCSC|C190271|[datasheet](libraries/parts/EEPROM_93LC56BT-I-OT/EEPROM_93LC56BT-I-OT.pdf)
USB-to-Serial|USB HS interface IC to dual UART/FIFO/SPI/JTAG/I2C|FTDI|FT2232HQ|Mouser|895-FT2232HQ|[datasheet](libraries/parts/USBtoSerial_FT2232HQ/USBtoSerial_FT2232HQ.pdf)
Ferrite Bead|GHz noise suppression SMD ferrite bead|Murata|BLM18HE152SN1D|LCSC|C82155|[datasheet](libraries/parts/FB_BLM18HE152SN1D/FB_BLM18HE152SN1D.pdf)
LDO 1.2V|Low dropout voltage regulator 800mA 1.2V|STM|LD1117S12TR|LCSC|C155612|[datasheet](libraries/parts/LDO_1v2_LD1117S12TR/LDO_1v2_LD1117S12TR.pdf)
LDO 3.3V|Low dropout voltage regulator 800mA 3.3V|Texas Instruments|LM1117IMPX-3.3|LCSC|C23984|[datasheet](libraries/parts/LDO_3v3_LM1117IMPX-3.3/LDO_3v3_LM1117IMPX-3.3.pdf)
RGB LED|Surface mount high-intensity RGB LED|Cree|CLMVC-FKA-CLBDGL7LBB79353|Mouser|941-CLMVCFKACLBDGL7L|[datasheet](libraries/parts/RGB_CLMVC-FKA-CLBDGL7LBB79353/RGB_CLMVC-FKA-CLBDGL7LBB79353.pdf)
FPGA|Lattice ICE40 UltraPlus 5280 LUTs 1.2V|Lattice|ICE40UP5K-SG48I|Mouser|842-ICE40UP5K-SG48I|[datasheet](libraries/parts/FPGA_ICE40UP5K-SG48I/FPGA_ICE40UP5K-SG48I.pdf)
NOR Flash|32M-bit 4Kb uniform sector SPI flash|WinBond|W25Q32JVSSIQ|LCSC|C179173|[datasheet](libraries/parts/NOR_Flash_W25Q32JVSSIQ/NOR_Flash_W25Q32JVSSIQ.pdf)
Oscillator|Clock oscillators 12MHz 1.6-3.6V|ECS|ECS-2520MV-120-BN-TR|Mouser|520-2520MV-120-BN-T|[datasheet](libraries/parts/Osc_12MHz_ECS-2520MV-120-BN-TR/Osc_12MHz_ECS-2520MV-120-BN-TR.pdf)
ADC|10-bit SPI 4 channel ADC|Microchip|MCP3004-I/SL|Mouser|579-MCP3004-I/SL|[datasheet](libraries/parts/ADC_MCP3004-I-SL/ADC_MCP3004-I-SL.pdf)
DAC|10-bit SPI 2 channel DAC|Microchip|MCP4812-E/SN|Mouser|579-MCP4812-E/SN|[datasheet](libraries/parts/DAC_MCP4812-E-SN/DAC_MCP4812-E-SN.pdf)
Pin Header|1x20 2.54 mm through-hole pin header|BOOMELE|C50981|LCSC|C50981|[datasheet](libraries/parts/PinHeader_C50981/PinHeader_C50981.pdf)
Diode|Fast switching diode: FV 0.8V @ 2.5 mA|Diodes Inc.|1N4148WS-7-F|LCSC|C60580|[datasheet](libraries/parts/Diode_1N4148WS-7-F/Diode_1N4148WS-7-F.pdf)
Fuse (resttable)|PTC 6V 750 mA trip|Bourns|MF-FSMF035X-2|LCSC|C116602|[datasheet](libraries/parts/PTC_MF-FSMF035X-2/PTC_MF-FSMF035X-2.pdf)
ESD Diode|ESD suppressors / TVS diodes|Nexperia|PRTR5V0U2F-115|LCSC|C478118|[datasheet](libraries/parts/ESD_Diodes_PRTR5V0U2F-115/ESD_Diodes_PRTR5V0U2F-115.pdf)
MOSEFT|SOT-723 N-Ch - 1.2 V gate|Rohm|RUM001L02T2CL|LCSC|C253528|[datasheet](libraries/parts/MOSFET_Nch_SOT-723/MOSFET_Nch_SOT-723_RUM001L02T2CL.pdf)
LED (Blue)|5 mA SMD 0402 Blue LED|Yongyu Photoelectric|SZYY0402B|LCSC|C434447|[datasheet](libraries/parts/LED_0402_SMD_Blue/LED_0402_SMD_Blue.pdf)
LED (Red)|5 mA SMD 0402 Red LED|Yongyu Photoelectric|SZYY0402R|LCSC|C434445|[datasheet](libraries/parts/LED_0402_SMD_Red/LED_0402_SMD_Red.pdf)
LED (White)|5 mA SMD 0402 White LED|Yongyu Photoelectric|SZYY0402W|LCSC|C434448|[datasheet](libraries/parts/LED_0402_SMD_Red/LED_0402_SMD_White.pdf)

## Fabrication

### Generate files
0. Save front/back renders and 3D model (step)
1. Create "fab" folder in KiCAD project root
2. Export BOM (as CSV): File->Fabrication outputs->BOM
3. Edit BOM: Open as ODS and save as XLSX in "fab" folder
4. Generate Gerbers: File->Fabrication outputs->Gerbers [settings](NB3_hindbrain_FAB_plot_settings.png)
5. Generate Drill: File->Fabrication outputs->Drill Files [settings](NB3_hindbrain_FAB_drill_settings.png)
6. Generate Centroids: File->Fabrication outputs->Componenet Placement [settings](NB3_hindbrain_FAB_pos_settings.png)
7. Zip entire fab folder (including BOM.xlsx)

### Upload boards(s) - PCBWay

#### Assembly Service
- Service: Turnkey Assembly
- Board type: Single Pieces
- Assembly: Both Sides
- Quantity: *2*

##### Other Parameters
- Number of unique Parts: 32
- Number of SMD Parts: 140
- Number of BGA/QFP Parts: 3
- Number of Through-Hole Parts: 2
			
#### PCB Specifications
- Board type: Single Pieces
- Size: 57 x 22.5 mm
- Quantity: 5
- Layer: 4 Layers (F In1 In2 B)
- Material: FR-4: TG150-160
- Thickness: 1.6 mm
- Min track/spacing: 5/5mil
- Min hole size: 0.25mm
- Solder mask: White
- Silkscreen: Black
- Edge connector: No
- Surface finish: Immersion gold(ENIG) (1U"), "HASL" to "ENIG" No
- Via process: Tenting vias
- Finished copper: 1 oz Cu
- Remove product No.: No
