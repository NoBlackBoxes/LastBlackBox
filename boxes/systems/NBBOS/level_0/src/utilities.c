#include "utilities.h"
#include "mailbox.h"

uint32_t get_clock_rate()
{
    mailbox[0] = 8*4; // Length of message in bytes
    mailbox[1] = MBOX_REQUEST;
    mailbox[2] = MBOX_TAG_GETCLKRATE; // Tag identifier
    mailbox[3] = 8; // Value size in bytes
    mailbox[4] = 0; // Value size in bytes
    mailbox[5] = 0x3; // Value
    mailbox[6] = 0; // Rate
    mailbox[7] = MBOX_TAG_LAST;

    if (mailbox_call(MBOX_CH_PROP)) {
        if (mailbox[5] == 0x3) {
           return mailbox[6];
        }
    }
    return 0;
}

uint32_t set_clock_rate(uint32_t rate)
{
    mailbox[0] = 9*4;  // Length of message in bytes
    mailbox[1] = MBOX_REQUEST;
    mailbox[2] = MBOX_TAG_SETCLKRATE; // Tag identifier
    mailbox[3] = 12;   // Value size in bytes
    mailbox[4] = 0;    // Value size in bytes
    mailbox[5] = 0x3;  // Value
    mailbox[6] = rate; // Rate
    mailbox[7] = 0;    // Rate
    mailbox[8] = MBOX_TAG_LAST;

    if (mailbox_call(MBOX_CH_PROP)) {
        if (mailbox[5] == 0x3 && mailbox[6] == rate) {
           return 1;
        }
    }
    return 0;
}
