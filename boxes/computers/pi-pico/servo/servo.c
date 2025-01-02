#include "pico/stdlib.h"
#include "hardware/pwm.h"
#include "hardware/clocks.h"

// Pico W devices use a GPIO on the WIFI chip for the LED,
// so when building for Pico W, CYW43_WL_GPIO_LED_PIN will be defined
#ifdef CYW43_WL_GPIO_LED_PIN
#include "pico/cyw43_arch.h"
#endif

#ifndef LED_DELAY_MS
#define LED_DELAY_MS 1000
#endif

#define ROTATE_0 544        //Rotate to 0° position (20000 = 50 ms)
#define ROTATE_180 2400

#define LL_CENTER 100
#define RL_CENTER 80

#define LF_CENTER 80
#define RF_CENTER 100

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

#define LEFT_LEG_PIN 10
#define RIGHT_LEG_PIN 12
#define LEFT_FOOT_PIN 11
#define RIGHT_FOOT_PIN 13

void test_limb(int limb, int center, int delay, int range);

int main() {
    int rc = pico_led_init();
    hard_assert(rc == PICO_OK);

    servo_setup(LEFT_LEG_PIN);
    servo_setup(RIGHT_LEG_PIN);
    servo_setup(LEFT_FOOT_PIN);
    servo_setup(RIGHT_FOOT_PIN);

    while (true) {
        pico_set_led(true);
        test_limb(LEFT_LEG_PIN, LL_CENTER, 10, 25);
        test_limb(RIGHT_LEG_PIN, RL_CENTER, 10, 25);
        test_limb(LEFT_FOOT_PIN, LF_CENTER, 10, 25);
        test_limb(RIGHT_FOOT_PIN, RF_CENTER, 10, 25);
        
        pico_set_led(false);
        sleep_ms(LED_DELAY_MS);
    }
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