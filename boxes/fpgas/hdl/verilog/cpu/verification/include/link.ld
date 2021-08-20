OUTPUT_ARCH("riscv")
ENTRY(test)
 
MEMORY
{
  rom (rx) : ORIGIN = 0x00000000, LENGTH = 4096
  ram (rw) : ORIGIN = 0x00000000, LENGTH = 4096
}
 
SECTIONS
{
  .text : { *(.text) } > rom
  .data : { *(.data) } > ram AT > rom
}