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

            // Initial: Set S=1, R=1
            t_s = 1'b1;
            t_r = 1'b1;

            // Set (S=0)
            #100 // 100 ns delay
            t_s = 1'b0;
            #100 // 100 ns delay
            t_s = 1'b1;

            // Reset (R=0)
            #100 // 100 ns delay
            t_r = 1'b0;
            #100 // 100 ns delay
            t_r = 1'b1;

            // Invalid (S=0, R=0)
            #100 // 100 ns delay
            t_s = 1'b0;
            t_r = 1'b0;
            #100 // 100 ns delay
            t_s = 1'b1;
            t_r = 1'b1;
            #100 // 100 ns delay
            t_s = 1'b0;
            t_r = 1'b0;

            // Reset (R=0)
            #100 // 100 ns delay
            t_s = 1'b1;
            t_r = 1'b1;
            #100 // 100 ns delay
            t_r = 1'b0;
            #100 // 100 ns delay
            t_r = 1'b1;

            // Wait
            #100 // 100 ns delay
            ;
        end
endmodule