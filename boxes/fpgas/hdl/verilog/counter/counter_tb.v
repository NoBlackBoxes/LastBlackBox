// Testbench for SR Latch
module counter_tb;

    // Declarations
    reg t_clock;
    reg t_reset;
    wire [7:0] t_left_leds;
    wire [7:0] t_right_leds;

    // Create instance of counter module
    counter test_counter(t_clock, t_reset, t_left_leds, t_right_leds);

    // Create clock
    always #5 t_clock = ~t_clock;   // 5 ns half-cycle

    // Test
    initial
        begin
            $dumpfile("bin/counter_tb.vcd");
            $dumpvars(0, counter_tb);
            $monitor(t_clock, t_reset, t_right_leds);

            // Initial
            t_clock <= 1'b0;
            t_reset <= 1'b1;
            #10 // 10 ns delay
            t_reset <= 1'b0;

            // Wait
            #200 // 200 ns delay
            ;

            // Final: Reset
            t_reset = 1'b1;
            #100 $finish; // 100 ns delay    
        end

endmodule