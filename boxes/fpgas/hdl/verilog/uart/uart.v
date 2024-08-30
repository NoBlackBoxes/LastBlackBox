// UART
module uart (clock, rx_pin, tx_pin, leds, gpio);

    // Declarations
    input wire  clock;          // Clock Input
    input wire  rx_pin;         // Serial Rx pin
    output wire tx_pin;         // Serial TX pin
    output wire [7:0] leds;     // Reporter LEDs
    output wire [5:0] gpio;     // Blanked GPIOs

    // Parameters
    parameter CLOCK_HZ = 12_000_000;
    parameter BAUD_RATE = 115200;

    // Internal
    wire rx_valid;
    wire [7:0] rx_byte;
    wire tx_valid;
    wire tx_done;
    wire [7:0] tx_byte;

    // Logic: Report
    assign gpio = 0;
    assign leds = rx_byte;

    // Logic: Echo
    assign tx_byte = rx_byte;
    assign tx_valid = rx_valid;

    // UART Receiver (Rx) module
    rx #(.BAUD_RATE(BAUD_RATE), .CLOCK_HZ(CLOCK_HZ)) uart_rx(
        .clock (clock),
        .pin   (rx_pin),
        .valid (rx_valid),
        .byte  (rx_byte)
    );

    // UART Transmitter (Tx) module
    tx #(.BAUD_RATE(BAUD_RATE), .CLOCK_HZ(CLOCK_HZ)) uart_tx(
        .clock (clock),
        .valid (tx_valid),
        .byte  (tx_byte),
        .done  (tx_done),
        .pin   (tx_pin) 
    );

endmodule