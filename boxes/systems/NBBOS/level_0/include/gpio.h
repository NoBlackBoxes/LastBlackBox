#pragma once
#include "common.h"

// Definitions (GPIO Registers)
#define GPIO_BASE               PERIPHERAL_BASE + 0x00200000
#define GPFSEL0                 GPIO_BASE + 0x00
#define GPFSEL1                 GPIO_BASE + 0x04
#define GPFSEL2                 GPIO_BASE + 0x08
#define GPFSEL3                 GPIO_BASE + 0x0c
#define GPFSEL4                 GPIO_BASE + 0x10
#define GPFSEL5                 GPIO_BASE + 0x14
#define GPSET0                  GPIO_BASE + 0x1c
#define GPSET1                  GPIO_BASE + 0x20
#define GPCLR0                  GPIO_BASE + 0x28
#define GPCLR1                  GPIO_BASE + 0x2c
#define GPLEV0                  GPIO_BASE + 0x34
#define GPLEV1                  GPIO_BASE + 0x38
#define GPEDS0                  GPIO_BASE + 0x40
#define GPEDS1                  GPIO_BASE + 0x44
#define GPREN0                  GPIO_BASE + 0x4c
#define GPREN1                  GPIO_BASE + 0x50
#define GPFEN0                  GPIO_BASE + 0x58
#define GPFEN1                  GPIO_BASE + 0x5c
#define GPHEN0                  GPIO_BASE + 0x64
#define GPHEN1                  GPIO_BASE + 0x68
#define GPLEN0                  GPIO_BASE + 0x70
#define GPLEN1                  GPIO_BASE + 0x74
#define GPAREN0                 GPIO_BASE + 0x7c
#define GPAREN1                 GPIO_BASE + 0x80
#define GPAFEN0                 GPIO_BASE + 0x88
#define GPAFEN1                 GPIO_BASE + 0x8c
#define GPIO_PUP_PDN_CNTRL_REG0 GPIO_BASE + 0xe4
#define GPIO_PUP_PDN_CNTRL_REG1 GPIO_BASE + 0xe8
#define GPIO_PUP_PDN_CNTRL_REG2 GPIO_BASE + 0xec
#define GPIO_PUP_PDN_CNTRL_REG3 GPIO_BASE + 0xf0

// Definitions (GPIO Parameters)
#define GPIO_MAXPIN             58
#define GPIO_LEDPIN             42

// Definitions (GPIO Functions)
#define GPIO_Function_Input     0
#define GPIO_Function_Output    1
#define GPIO_Function_Alt0      4
#define GPIO_Function_Alt1      5
#define GPIO_Function_Alt2      6
#define GPIO_Function_Alt3      7
#define GPIO_Function_Alt4      3
#define GPIO_Function_Alt5      2

// Definitions (GPIO Pull States)
#define GPIO_Pull_None          0
#define GPIO_Pull_Up            1
#define GPIO_Pull_Down          2
#define GPIO_Pull_Reserved      3

// Declarations
void gpio_set_function(uint32_t pin, uint32_t function);
void gpio_set_pull(uint32_t pin, uint32_t pull);
void gpio_pin_set(uint32_t pin);
void gpio_pin_clear(uint32_t pin);
