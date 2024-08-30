// Testbench for ADC
`timescale 1ns/10ps
module adc_tb;

    // Declarations
    reg  t_clock;           // Clock Input
    reg  t_D_out;           // ADC serial data
    wire t_sample_clock;    // ADC Clock
    wire t_D_in;            // ADC control data
    reg  t_rx_pin;          // Serial Rx pin
    wire [7:0] t_leds;      // Reporter LEDs
    wire [5:0] t_gpio;      // Blanked GPIOs

    // Constants
    localparam SAMPLE_CLOCK_DIV = 24'h0000FF;
    localparam CLOCK_HZ = 12_000_000;
    localparam CLOCK_PERIOD_NS = 1_000_000_000 / CLOCK_HZ;
    localparam BAUD_RATE = 115200;
    localparam CYCLES_PER_BIT = CLOCK_HZ / BAUD_RATE;           // Clock cycles per bit
    localparam BIT_PERIOD = CLOCK_PERIOD_NS * CYCLES_PER_BIT;

    // Create instance of ADC module
    adc #(.SAMPLE_CLOCK_DIV(SAMPLE_CLOCK_DIV), .BAUD_RATE(BAUD_RATE), .CLOCK_HZ(CLOCK_HZ)) test_adc(
        .clock          (t_clock),
        .D_out          (t_D_out),
        .sample_clock   (t_sample_clock),
        .D_in           (t_D_in),
        .tx_pin         (t_tx_pin),
        .leds           (t_leds),
        .gpio           (t_gpio)
    );

    // Create clock
    always #50 t_clock = ~t_clock;   // 50 ns half-cycle (10 MHz)

    // Test
    initial
        begin
            $dumpfile("bin/adc_tb.vcd");
            $dumpvars(0, adc_tb);

            // Initial
            t_clock <= 1'b0;

            // ADC Conversion
            #100000000; // 100 ms delay

            // Finish
            $finish;
        end

endmodule