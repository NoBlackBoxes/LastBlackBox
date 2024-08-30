#pragma once
#include "common.h"

// Definitions (Mailbox Registers)
#define VIDEOCORE_MBOX  PERIPHERAL_BASE + 0x0000B880
#define MBOX_READ       VIDEOCORE_MBOX + 0x0
#define MBOX_POLL       VIDEOCORE_MBOX + 0x10
#define MBOX_SENDER     VIDEOCORE_MBOX + 0x14
#define MBOX_STATUS     VIDEOCORE_MBOX + 0x18
#define MBOX_CONFIG     VIDEOCORE_MBOX + 0x1C
#define MBOX_WRITE      VIDEOCORE_MBOX + 0x20

// Definitions (Mailbox Codes)
#define MBOX_RESPONSE   0x80000000
#define MBOX_FULL       0x80000000
#define MBOX_EMPTY      0x40000000

enum {
    MBOX_REQUEST  = 0
};

enum {
    MBOX_CH_POWER = 0,
    MBOX_CH_FB    = 1,
    MBOX_CH_VUART = 2,
    MBOX_CH_VCHIQ = 3,
    MBOX_CH_LEDS  = 4,
    MBOX_CH_BTNS  = 5,
    MBOX_CH_TOUCH = 6,
    MBOX_CH_COUNT = 7,
    MBOX_CH_PROP  = 8 // Request from ARM for response by VideoCore
};

enum {
    MBOX_TAG_SETPOWER   = 0x28001,
    MBOX_TAG_GETCLKRATE = 0x30002,
    MBOX_TAG_GETCLKMAXM = 0x30004,
    MBOX_TAG_SETCLKRATE = 0x38002,

    MBOX_TAG_SETPHYWH   = 0x48003,
    MBOX_TAG_SETVIRTWH  = 0x48004,
    MBOX_TAG_SETVIRTOFF = 0x48009,
    MBOX_TAG_SETDEPTH   = 0x48005,
    MBOX_TAG_SETPXLORDR = 0x48006,
    MBOX_TAG_GETFB      = 0x40001,
    MBOX_TAG_GETPITCH   = 0x40008,

    MBOX_TAG_LAST       = 0
};

// Global
extern volatile uint32_t mailbox[36];

// Declarations
uint32_t mailbox_call(uint8_t ch);
