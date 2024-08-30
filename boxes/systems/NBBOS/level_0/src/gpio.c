#include "common.h"
#include "gpio.h"

// Set GPIO pin function
void gpio_set_function(uint32_t pin, uint32_t function)
{
    uint64_t address = GPFSEL0 + ((pin / 10) * 4);  // Determine function selection register (groups of 10 pins)
    int32_t bit_offset = (pin * 3) % 30;            // Determine bit offset of pin within bank
    uint32_t current_value = GET32(address);        // Retrieve current register value
    current_value &= ~(7 << bit_offset);            // Clear 3 bits at offset
    current_value |= (function << bit_offset);      // Store new bits at offset
    PUT32(address, current_value);                  // Set selection register

    return;
}

// Set GPIO pin pull resistor
void gpio_set_pull(uint32_t pin, uint32_t pull)
{
    uint64_t address = GPIO_PUP_PDN_CNTRL_REG0 + ((pin / 16) * 4);    // Determine pull selection register (groups of 16 pins)
    int32_t bit_offset = (pin * 2) % 32;                                        // Determine bit offset of pin within bank
    uint32_t current_value = GET32(address);                 // Retrieve current register value
    current_value &= ~(3 << bit_offset);                                    // Clear 2 bits at offset
    current_value |= (pull << bit_offset);                                  // Store new bits at offset
    PUT32(address, current_value);                               // Set selection register

    return;
}

// Set (True)
void gpio_pin_set(uint32_t pin) 
{
    if(pin < 32)
    {
        PUT32(GPSET0, (1 << pin));    
    }
    else if(pin < GPIO_MAXPIN)
    {
        PUT32(GPSET1, (1 << (pin - 32)));    
    }    
    return;
}

// Clear (False)
void gpio_pin_clear(uint32_t pin) 
{
    if(pin < 32)
    {
        PUT32(GPCLR0, (1 << pin));    
    }
    else if(pin < GPIO_MAXPIN)
    {
        PUT32(GPCLR1, (1 << (pin - 32)));    
    }    
    return;
}
