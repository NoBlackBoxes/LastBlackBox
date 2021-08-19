OUTPUT_ARCH("riscv")
 
ENTRY(test)
 
MEMORY
{
  rom (rxai!w) : ORIGIN = 0x0, LENGTH = 4K
  ram (wxa!ri) : ORIGIN = 0x1FFFF, LENGTH = 4K
}
 
SECTIONS
{
  . = 0x0;
  .text : { *(.text) }
  . = 0x1FFFF;
  .data : { *(.data) }
  .bss : { *(.bss) }
}