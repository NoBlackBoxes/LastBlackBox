// Testbench for SR Latch
module sr_latch_tb;

    // Declarations
    reg t_s;
    reg t_r;
    wire t_q;
    wire t_qn;

    // Create instance of sr_latch module
    sr_latch test_sr_latch(t_s, t_r, t_q, t_qn);

    // Test
    initial
        begin
            $dumpfile("bin/sr_latch_tb.vcd");
            $dumpvars(0, sr_latch_tb);
            $monitor(t_s, t_r, t_q, t_qn);

            // Initial (1, 1)
            t_s = 1'b0;
            t_r = 1'b0;

            // Set (1, 0)
            #5 // 5 ns delay
            t_s = 1'b1;

            // Reset (x, 1)
            #5 // 5 ns delay
            t_r = 1'b0;

            // Fiddle (1, 0)
            #5 // 5 ns delay
            t_s = 1'b1;
            t_r = 1'b0;

            // Set (1, 1)
            #5 // 5 ns delay
            t_s = 1'b1;
            t_r = 1'b1;

            // Fiddle (0, 0)
            #5 // 5 ns delay
            t_s = 1'b0;
            t_r = 1'b0;

            // Set (0, 1)
            #5 // 5 ns delay
            t_s = 1'b0;
            t_r = 1'b1;

            // Fiddle (1, 0)
            #5 // 5 ns delay
            t_s = 1'b1;
            t_r = 1'b0;

        end
endmodule