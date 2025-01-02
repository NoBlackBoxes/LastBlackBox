#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/i2c.h"
#include "pico/binary_info.h"

#define HT16K33_SYSTEM_STANDBY 0x20
#define HT16K33_SYSTEM_RUN 0x21
#define HT16K33_SET_ROW_INT 0xA0
#define HT16K33_BRIGHTNESS 0xE0

// Display on/off/blink
#define HT16K33_DISPLAY_SETUP 0x80
#define HT16K33_DISPLAY_RAM 0x00

// OR/clear these to display setup register
#define HT16K33_DISPLAY_OFF 0x0
#define HT16K33_DISPLAY_ON 0x1
#define HT16K33_BLINK_OFF 0x0
#define HT16K33_BLINK_2HZ 0x2
#define HT16K33_BLINK_1HZ 0x4
#define HT16K33_BLINK_0p5HZ 0x6

const int I2C_addr = 0x71;

// Function Declarations
void init();
void i2c_write_byte(uint8_t val);
void i2c_set_brightness(int brightness);
void i2c_clear();

// Function Definitions
void init()
{
    i2c_write_byte(HT16K33_SYSTEM_RUN);
    i2c_write_byte(HT16K33_SET_ROW_INT);
    i2c_write_byte(HT16K33_DISPLAY_SETUP | HT16K33_DISPLAY_ON);
}

void i2c_write_byte(uint8_t val)
{
    i2c_write_blocking(i2c_default, I2C_addr, &val, 1, false);
}

void i2c_set_brightness(int brightness)
{
    i2c_write_byte(HT16K33_BRIGHTNESS | (brightness <= 15 ? brightness : 15));
}

void i2c_clear()
{
    uint8_t command[17];
    command[0] = HT16K33_DISPLAY_RAM;
    for (unsigned char row = 1; row < 17; row++)
    {
        command[row] = 0x00;
    }
    i2c_write_blocking(i2c_default, I2C_addr, command, 17, false);
}

void i2c_row(int row)
{
    uint8_t command[17];
    command[0] = HT16K33_DISPLAY_RAM;
    for (unsigned char col = 1; col < 17; col++)
    {

        command[col] = 1 << row;
    }
    i2c_write_blocking(i2c_default, I2C_addr, command, 17, false);
}

int main()
{
    stdio_init_all();

    // This example will use I2C0 on the default SDA and SCL pins (4, 5 on a Pico)
    i2c_init(i2c_default, 100 * 1000);
    gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
    gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);
    // Make the I2C pins available to picotool
    bi_decl(bi_2pins_with_func(PICO_DEFAULT_I2C_SDA_PIN, PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C));
    printf("Welcome to Matrix!\n");

    init();
    i2c_set_brightness(0);
    int count = 0;
    while (true)
    {
        //i2c_clear();
        i2c_row(count);
        count = (count + 1) % 8;
        printf("Hello, world!\n");
        sleep_ms((count+10)*10);
    }
}