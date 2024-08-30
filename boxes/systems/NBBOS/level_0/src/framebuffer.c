#include "common.h"
#include "mailbox.h"
#include "uart.h"

uint32_t width, height, pitch, isrgb;
uint32_t *framebuffer;

void framebuffer_init()
{
    mailbox[0] = 35*4; // Length of message in bytes
    mailbox[1] = MBOX_REQUEST;

    mailbox[2] = MBOX_TAG_SETPHYWH; // Tag identifier
    mailbox[3] = 8; // Value size in bytes
    mailbox[4] = 0;
    mailbox[5] = 1920; // Value(width)
    mailbox[6] = 1080; // Value(height)

    mailbox[7] = MBOX_TAG_SETVIRTWH;
    mailbox[8] = 8;
    mailbox[9] = 8;
    mailbox[10] = 1920;
    mailbox[11] = 1080;

    mailbox[12] = MBOX_TAG_SETVIRTOFF;
    mailbox[13] = 8;
    mailbox[14] = 8;
    mailbox[15] = 0; // Value(x)
    mailbox[16] = 0; // Value(y)

    mailbox[17] = MBOX_TAG_SETDEPTH;
    mailbox[18] = 4;
    mailbox[19] = 4;
    mailbox[20] = 32; // Bits per pixel

    mailbox[21] = MBOX_TAG_SETPXLORDR;
    mailbox[22] = 4;
    mailbox[23] = 4;
    mailbox[24] = 1; // RGB

    mailbox[25] = MBOX_TAG_GETFB;
    mailbox[26] = 8;
    mailbox[27] = 8;
    mailbox[28] = 4096; // FrameBufferInfo.pointer
    mailbox[29] = 0;    // FrameBufferInfo.size

    mailbox[30] = MBOX_TAG_GETPITCH;
    mailbox[31] = 4;
    mailbox[32] = 4;
    mailbox[33] = 0; // Bytes per line

    mailbox[34] = MBOX_TAG_LAST;

    // Check call is successful and we have a pointer with depth 32
    uint32_t result = mailbox_call(MBOX_CH_PROP);
    if (result && mailbox[20] == 32 && mailbox[28] != 0) {
        mailbox[28] &= 0x3FFFFFFF; // Convert GPU address to ARM address
        width = mailbox[10];       // Actual physical width
        height = mailbox[11];      // Actual physical height
        pitch = mailbox[33];       // Number of bytes per line
        isrgb = mailbox[24];       // Pixel order
        framebuffer = (uint32_t *)((uint64_t)mailbox[28]);
        uart_send_string("Got a framebuffer\n");
    }
    else
    {
        uart_send_string("Failed to get framebuffer\n");
    }
}

void framebuffer_clear()
{
    //for (uint32_t i = 0; i < 1920*1080; i++)
    //{
    //    framebuffer[i] = 0xFF000000;
    //}
    __memset_aarch64(framebuffer,  0xFF000000, 1920*1080*4);
}

void framebuffer_fill_naive(char r, char g, char b)
{
    uint32_t color = (0xFF << 24) | (r << 16) | (g << 8) | (b);
    for (uint32_t i = 0; i < 1920*1080; i++)
    {
        framebuffer[i] = color;
    }
}

void framebuffer_fill(char r, char g, char b)
{
    uint32_t color = (0xFF << 24) | (r << 16) | (g << 8) | (b);
    __memset_aarch64(framebuffer,  color, 1920*1080*4);
}
