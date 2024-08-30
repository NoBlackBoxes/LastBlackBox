// Testbench for UART
`timescale 1ns/10ps
module uart_tb;

    // Declarations
    reg  t_clock;           // Clock Input
    reg  t_rx_pin;          // Serial Rx pin
    wire t_tx_pin;          // Serial TX pin
    wire t_rx_valid;
    wire [7:0] t_rx_byte;
    reg  [7:0] t_rx_data;
    reg  t_tx_valid;
    wire t_tx_done;
    reg  [7:0] t_tx_byte;
    integer i;

    // Constants
    localparam CLOCK_HZ = 10_000_000;
    localparam CLOCK_PERIOD_NS = 1_000_000_000 / CLOCK_HZ;
    localparam BAUD_RATE = 115200;
    localparam CYCLES_PER_BIT = CLOCK_HZ / BAUD_RATE;           // Clock cycles per bit
    localparam BIT_PERIOD = CLOCK_PERIOD_NS * CYCLES_PER_BIT;

    // Create clock
    always #50 t_clock = ~t_clock;   // 50 ns half-cycle (10 MHz)

    // Test
    initial
        begin
            $dumpfile("bin/uart_tb.vcd");
            $dumpvars(0, uart_tb);
            //$monitor(t_clock, t_rx_pin, t_tx_pin);

            // Initial
            t_clock <= 1'b0;
            t_tx_valid <= 1'b0;
            #10 // 10 ns delay

            // Transmit byte
            @(posedge t_clock);
            t_tx_valid <= 1'b1;
            t_tx_byte <= 8'hAD;
            @(posedge t_clock);
            t_tx_valid <= 1'b0;
            @(posedge t_tx_done);

            // Receive byte
            @(posedge t_clock);
            t_rx_data <= 8'hB7;
            begin
                t_rx_pin <= 1'b0;                   // Send START bit
                #(BIT_PERIOD);                      // Wait one bit period
                for (i = 0; i < 8; i = i+1)
                    begin
                        t_rx_pin <= t_rx_data[i];   // Send Data Byte
                        #(BIT_PERIOD);
                    end
                t_rx_pin <= 1'b1;                   // Send STOP Bit
                #(BIT_PERIOD);
            end  

            // Tests
            if (t_rx_byte == 8'hB7)
                begin
                    $display("%c[1;32m",27);
                    $display("Test Passed - Correct Byte Received");
                    $display("%c[0m",27);
                end
            else
                begin
                    $display("%c[1;31m",27);
                    $display("Test Failed - Incorrect Byte Received");
                    $display("%c[0m",27);
                end

            // Final
            #100 $finish; // 100 ns delay
        end

    // UART Receiver (Rx) module
    rx #(.BAUD_RATE(BAUD_RATE), .CLOCK_HZ(CLOCK_HZ)) uart_rx(
        .clock (t_clock),
        .pin   (t_rx_pin),
        .valid (t_rx_valid),
        .byte  (t_rx_byte)
    );

    // UART Transmitter (Tx) module
    tx #(.BAUD_RATE(BAUD_RATE), .CLOCK_HZ(CLOCK_HZ)) uart_tx(
        .clock (t_clock),
        .valid (t_tx_valid),
        .byte  (t_tx_byte),
        .done  (t_tx_done),
        .pin   (t_tx_pin) 
    );
  
  endmodule