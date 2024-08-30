// Blinks LED connected to Pin ??
#include <avr/io.h>
#include <util/delay.h>

int main(void)
{
    DDRB = 0b00100000; // Set DDR (data direction register) of PORT B to output for LED bit only

    while (1)
    {
        // LED on
        PORTB = 0b00100000; // Set LED bit to high
        _delay_ms(500);     // Wait 500 milliseconds

        //LED off
        PORTB = 0b00000000; // Set LED bit to low
        _delay_ms(500);     // Wait 500 milliseconds
    }
}

//FIN