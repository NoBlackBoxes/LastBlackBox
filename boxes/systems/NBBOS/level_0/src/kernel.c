#include "common.h"
#include "utilities.h"
#include "uart.h"
#include "gpio.h"
#include "framebuffer.h"

void kernel_main()
{
    // Initialize UART
    uart_init();

    // Say hello
    uart_send_string("Hello Everybody!\n");

    // Update CPU Clocks (this sequence breaks for O3 and O4 compiliation)
    uint32_t clock_rate = get_clock_rate();
    uart_report("Initial Clock Rate (Hz)", clock_rate);
    set_clock_rate(1500000000);
    clock_rate = get_clock_rate();
    uart_report("New Clock Rate (Hz)", clock_rate);

    // Get current exception level
    uint32_t el = GETEL();
    uart_report("Current Exception Level", el);

    // Initialize framebuffer
    framebuffer_init();

    // Select GPIO functions
	gpio_set_function(GPIO_LEDPIN, GPIO_Function_Output);
	gpio_set_function(16, GPIO_Function_Output);

    int rg = 0;
    int r,g,b;
    r = 0;
    g = 55;
    b = 77;
    int dr = 2;
    int dg = 5;
    int db = 11;

    while (1)
    {
        //framebuffer_fill_naive(r,g,b);
        framebuffer_fill(r,g,b);
        r += dr;
        g += dg;
        b += db;
        if(r > 255 || r < 0)
        {
            dr = -dr;
            r += dr;
        }
        if(g > 255 || g < 0)
        {
            dg = -dg;
            g += dg;
        }
        if(b > 255 || b < 0)
        {
            db = -db;
            b += db;
        }

        if (rg == 1)
        {
            gpio_pin_set(GPIO_LEDPIN);
            gpio_pin_set(16);
            rg = 0;
        }
        else
        {
            gpio_pin_clear(GPIO_LEDPIN);
            gpio_pin_clear(16);
            rg = 1;
        }
        // TICK(10000);
    }
}
