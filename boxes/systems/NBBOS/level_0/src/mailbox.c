#include "common.h"
#include "mailbox.h"
#include "uart.h"

// The buffer must be 16-byte aligned as only the upper 28 bits of the address can be passed via the mailbox
volatile uint32_t __attribute__((aligned(16))) mailbox[36];

// Declarations
uint32_t mailbox_call(uint8_t ch)
{
    // 28-bit address (MSB) and 4-bit value (LSB)
    uint64_t mailbox_address = (uint64_t) &mailbox; 
    uint32_t r = (((uint32_t) mailbox_address) &~ 0xF) | (ch & 0xF);

    // Wait until we can write
    while (GET32(MBOX_STATUS) & MBOX_FULL);
    
    // Write the address of our buffer to the mailbox with the channel appended
    PUT32(MBOX_WRITE, r);

    while (1) {
        // Is there a reply?
        uint32_t status = MBOX_EMPTY;
        while (status & MBOX_EMPTY)
        {
            status = GET32(MBOX_STATUS);
        }

        // Is it a reply to our message?
        uint32_t re = GET32(MBOX_READ);
        if (r == re)
        {
            return mailbox[1]==MBOX_RESPONSE; // Is it successful?
        }
    }
    return 0;
}
