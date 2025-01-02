#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/binary_info.h"
#include "hardware/pwm.h"
#include "hardware/clocks.h"
#include "hardware/i2c.h"

// Pico W devices use a GPIO on the WIFI chip for the LED,
// so when building for Pico W, CYW43_WL_GPIO_LED_PIN will be defined
#ifdef CYW43_WL_GPIO_LED_PIN
#include "pico/cyw43_arch.h"
#endif

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


#define LEFT_LEG_PIN 10
#define RIGHT_LEG_PIN 12
#define LEFT_FOOT_PIN 11
#define RIGHT_FOOT_PIN 13


#ifndef LED_DELAY_MS
#define LED_DELAY_MS 1000
#endif

#define ROTATE_0 544        //Rotate to 0° position (20000 = 50 ms)
#define ROTATE_180 2400

#define LL_CENTER 100
#define RL_CENTER 80

#define LF_CENTER 80
#define RF_CENTER 100

// Function Declarations
int pico_led_init(void);
void pico_set_led(bool led_on);

void servo_setup(int pin);
void servo_go(int pin, float degree);
void test_limb(int limb, int center, int delay, int range);

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


// Perform initialisation
int pico_led_init(void) {
#if defined(PICO_DEFAULT_LED_PIN)
    // A device like Pico that uses a GPIO for the LED will define PICO_DEFAULT_LED_PIN
    // so we can use normal GPIO functionality to turn the led on and off
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    return PICO_OK;
#elif defined(CYW43_WL_GPIO_LED_PIN)
    // For Pico W devices we need to initialise the driver etc
    return cyw43_arch_init();
#endif
}

// Servo setup
void servo_setup(int pin)
{
	gpio_init(pin);

	//Setup up PWM t
	gpio_set_function(pin, GPIO_FUNC_PWM);
	pwm_set_gpio_level(pin, 0);
	uint slice_num = pwm_gpio_to_slice_num(pin);

	// Get clock speed and compute divider for 50 hz
	uint32_t clk = clock_get_hz(clk_sys);
	uint32_t div = clk / (20000 * 50);

	// Check div is in range
	if ( div < 1 ){
		div = 1;
	}
	if ( div > 255 ){
		div = 255;
	}

	pwm_config config = pwm_get_default_config();
	pwm_config_set_clkdiv(&config, (float)div);

	// Set wrap so the period is 20 ms
	pwm_config_set_wrap(&config, 20000);

	// Load the configuration
	pwm_init(slice_num, &config, false);

	pwm_set_enabled(slice_num, true);

}

void servo_go(int pin, float degree)
{
    if (degree > 180.0){
        return;
	}
	if (degree < 0){
		return;
	}

	int duty = (((float)(ROTATE_180 - ROTATE_0) / 180.0) * degree) + ROTATE_0;

	//printf("PWM for %f deg is %d duty\n", degree, duty);
	pwm_set_gpio_level(pin, duty);
}


// Turn the led on or off
void pico_set_led(bool led_on) {
#if defined(PICO_DEFAULT_LED_PIN)
    // Just set the GPIO on or off
    gpio_put(PICO_DEFAULT_LED_PIN, led_on);
#elif defined(CYW43_WL_GPIO_LED_PIN)
    // Ask the wifi "driver" to set the GPIO on or off
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, led_on);
#endif
}

void test_limb(int limb, int center, int delay, int range)
{
        for(int i = -range; i < range; i++)
        {
            servo_go(limb, center + i);
            sleep_ms(delay);
        }
        for(int i = range; i > -range; i--)
        {
            servo_go(limb, center + i);
            sleep_ms(delay);
        }
        servo_go(limb, center);

        return;
}

int main() {

    stdio_init_all();

    // This example will use I2C0 on the default SDA and SCL pins (4, 5 on a Pico)
    i2c_init(i2c_default, 100 * 1000);
    gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
    gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);
    // Make the I2C pins available to picotool
    bi_decl(bi_2pins_with_func(PICO_DEFAULT_I2C_SDA_PIN, PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C));
    printf("Welcome to Walking Robot!\n");

    init();
    i2c_set_brightness(0);

    int rc = pico_led_init();
    hard_assert(rc == PICO_OK);

    servo_setup(LEFT_LEG_PIN);
    servo_setup(RIGHT_LEG_PIN);
    servo_setup(LEFT_FOOT_PIN);
    servo_setup(RIGHT_FOOT_PIN);

    int count = 0;
    while (true) {
        i2c_row(count);
        count = (count + 1) % 8;
        pico_set_led(true);
        test_limb(LEFT_LEG_PIN, LL_CENTER, 10, 25);
        i2c_row(count);
        count = (count + 1) % 8;
        test_limb(RIGHT_LEG_PIN, RL_CENTER, 10, 25);
        i2c_row(count);
        count = (count + 1) % 8;
        test_limb(LEFT_FOOT_PIN, LF_CENTER, 10, 25);
        i2c_row(count);
        count = (count + 1) % 8;
        test_limb(RIGHT_FOOT_PIN, RF_CENTER, 10, 25);
        i2c_row(count);
        count = (count + 1) % 8;
        
        pico_set_led(false);
        sleep_ms(LED_DELAY_MS);
    }
}
