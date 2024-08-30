#pragma once

// Include STD headers
#include <stdint.h>

// Definitions (Peripheral base address)
#define PERIPHERAL_BASE 0xFE000000

// External declarations (ASM utilities)
extern void PUT32 (uint64_t address, uint32_t value);
extern uint32_t GET32 (uint64_t address);
extern void TICK (uint64_t count);
extern uint32_t GETEL (void);
extern void __memset_aarch64(uint32_t *dst, uint32_t val, uint32_t count);