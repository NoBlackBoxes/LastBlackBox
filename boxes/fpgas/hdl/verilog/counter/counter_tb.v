// Testbench for SR Latch
module counter_tb;

    // Declarations
    reg t_clock;
    reg t_reset;
    wire [3:0] t_count;

    // Create instance of counter module
    counter test_counter(t_clock, t_reset, t_count);

    // Create clock
    always #5 t_clock = ~t_clock;

    // Test
    initial
        begin
            $dumpfile("bin/counter_tb.vcd");
            $dumpvars(0, counter_tb);
            $monitor(t_clock, t_reset, t_count);

            // Initial
            t_clock <= 1'b0;
            t_reset <= 1'b1;

            // Initial: Reset
            t_reset <= 1'b0;
            #10 // 100 ns delay
            t_reset <= 1'b1;

            // Wait
            #200 // 100 ns delay
            ;

            // Final: Reset
            t_reset = 1'b0;
            #10 // 100 ns delay
            t_reset = 1'b1;

            // Finish
            #100 $finish; // 100 ns delay    
        end

endmodule