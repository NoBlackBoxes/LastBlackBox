.section ".text.boot"

// The "start" of NBBOS
.globl _start
_start:
    // Check that we are on the main core
    mrs x0, mpidr_el1       // Load the core ID into x0
    and x0, x0, #0xFF       // Isolate the lowest byte
    cbz x0, setup           // If zero, then we are on the main core (branch to setup)
    b hang                  // If not, then hang

// Setup and clear the stack
setup:
    // Set the stack pointer
    ldr x1, =_start         // Load the address of _start in x1
    mov sp, x1              // Set the stack pointer to this address (stack will grow down)

    // Clear the BSS (stack) region
    ldr x1, =__bss_start    // Load first address of BSS into x0
    ldr w2, =__bss_size     // Load size of BSS into w2
1:  cbz w2, mmu             // Check if w2 is zero, if so setup MMU
    str xzr, [x1], #8       // Store zero (the xzr register, 64-bits/8-bytes) at address in x1 and then increment x1 by 8
    sub w2, w2, #1          // Decrement w2
    cbnz w2, 1b             // Loop (go to backward reference 1, 1b) if result w2 not zero

mmu:
    // Initialize translation table control registers
    ldr x1, =0x3520         // 0x3520 = (00) 00 11 01 01 00 100000
                            // 18:16 - Physical address size (00): 4GB space
                            // 15:14 - Granule size 00: 4 kB granularity 
                            // 13:12 - Sharability 11: Inner sharable
                            // 11:10 - Outer cachability 01: Normal memory, outer cacheable
                            // 9:8   - Inner cachability 01: Normal memory, inner cacheable
                            // 7:6   - Resevered 00: 
                            // 5:0   - Size offset 2^(64-32) 100000: 2^32
    msr tcr_el2, x1         // Set TCR_EL2 system register
    ldr x1, =0xFF440400     // 0xFF440400 = 11111111 01000100 00000100 00000000
                            // ATTR0 (00000000) Device-nGnRnE
                            // ATTR1 (00000100) Device-nGnRE
                            // ATTR2 (01000100) Normal Non-Cacheable
                            // ATTR3 (11111111) Normal inner/outer write-back (cacheable), alllocate RW
    msr mair_el2, x11       // Set MAIR_EL2 system register
    adr x0, ttb0_base       // Store adreess of ttb0_base, must be a 4KB-aligned address.
    msr ttbr0_el2, x0       // Set TTBR0_EL2 system register

    // Set up translation table entries in memory with looped store instructions
    // Set the level 1 translation table
    
    // The first entry points to level2_pagetable.
    ldr x1, =level2_pagetable   // Must be a 4KB align address.
    ldr x2, =0xFFFFF000
    and x2, x1, x2              // NSTable=0 APTable=0 XNTable=0 PXNTable=0.
    orr x2, x2, 0x3
    str x2, [x0], #8

    // The second entry is 1GB block from 0x40000000 to 0x7FFFFFFF.
    ldr x2, =0x40000741         // Executable Inner and Outer Shareable.
    str x2, [x0], #8            // R/W at all ELs secure memory
                                // AttrIdx=000 Device-nGnRnE.
    
    // The third entry is 1GB block from 0x80000000 to 0xBFFFFFFF.
    ldr x2, =0x80000741
    str x2, [x0], #8

    // The fourth entry is 1GB block from 0xC0000000 to 0xFFFFFFFF.
    ldr x2, =0xC0000741
    str x2, [x0], #8

    // Set level 2 translation table.
    ldr x0, =level2_pagetable   // Base address of level2_pagetable.
    ldr x2, =0x0000074D         // Executable Inner and Outer Shareable.
                                // R/W at all ELs secure memory.
                                // AttrIdx=011 Normal Cacheable.
    mov x4, #512                // Set 512 level2 block entries.
    ldr x5, =0x00200000         // Increase 2MB address each time.
loop:
    str x2, [x0], #8            // Each entry occupies 2 words.
    add x2, x2, x5
    subs x4, x4, #1
    bne loop
    b cache

cache:
    // Set the SMPEN bit
    mrs x0, S3_1_C15_C2_1
    orr x0, x0, #(0x1 << 6)     // The SMP bit.
    msr S3_1_C15_C2_1, x0

    // Turn on Cache and MMU
	mrs	x0, sctlr_el2
	orr x0, x0, #(0x1 << 2)     // The C bit on (data cache). 
	orr x0, x0, #(0x1 << 12)    // The I bit on (instruction cache)
	orr x0, x0, #(0x1 << 0)     // Enable MMU (this breaks)
	msr	sctlr_el2, x0
    dsb sy
    isb
    b code

code:
    // Jump to code entry point
    bl kernel_main

    // Hang if our code returns
    b  hang

// Hang (infinite loop)
hang:
    wfe                     // Wait for event (low-power)
    b hang


// Put a 64-bit value with little endianness.
.macro PUT_64B high, low
.word \low
.word \high
.endm

// Create an entry pointing to a next-level table.
.macro TABLE_ENTRY PA, ATTR
PUT_64B \ATTR, (\PA) + 0x3
.endm

// Create an entry for a 1GB block.
.macro BLOCK_1GB PA, ATTR_HI, ATTR_LO
PUT_64B \ATTR_HI, ((\PA) & 0xC0000000) | \ATTR_LO | 0x1
.endm

// Create an entry for a 2MB block.
.macro BLOCK_2MB PA, ATTR_HI, ATTR_LO
PUT_64B \ATTR_HI, ((\PA) & 0xFFE00000) | \ATTR_LO | 0x1
.endm

.align 12 // 12 for 4KB granule.
ttb0_base:
TABLE_ENTRY level2_pagetable, 0
BLOCK_1GB 0x40000000, 0, 0x740
BLOCK_1GB 0x80000000, 0, 0x740
BLOCK_1GB 0xC0000000, 0, 0x740

.align 12 // 12 for 4KB granule.
level2_pagetable:
.set ADDR, 0x000 // The current page address.
.rept 0x200
BLOCK_2MB (ADDR << 20), 0, 0x74C
.set ADDR, ADDR+2
.endr