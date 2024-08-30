// Reads a digital input connected to Pin PD2 and toggles LED connected to Pin PB5
#include <avr/io.h>
#include <util/delay.h>

// Declare functions
void setup();

// Main function
int main(void)
{
    // Run setup
    setup();

    while (1)
    {
        // Read pin (PD2) and toggle output LED (PB5)
        if (PIND & (1 << PD2))
        {
            PORTB |= (1 << PB5);
        }
        else
        {
            PORTB &= ~(1 << PB5);
        }
    }
}

// Setup function
void setup(void)
{
    DDRB = 0b00100000;   // Set DDR (data direction register) of PORT B to output for LED bit only
    DDRD &= ~(1 << PD2); // Set PD2 data direction to input
}
//FIN