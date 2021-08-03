// Testbench for Flopenr
module flopenr_tb;

    // Declarations
    reg t_clock;
    reg t_reset;
    reg t_enable;
    reg [7:0] t_d;
    wire [7:0] t_q;

    // Create instance of flopenr module
    defparam test_flopenr.WIDTH = 8;
    flopenr test_flopenr(t_clock, t_reset, t_enable, t_d, t_q);

    // Test
    initial
        begin
            $dumpfile("bin/flopenr_tb.vcd");
            $dumpvars(0, flopenr_tb);
            $monitor(t_clock, t_reset, t_enable, t_d, t_q);

            // Initial
            t_clock <= 1'b0;
            t_reset <= 1'b0;
            t_enable <= 1'b0;
            t_d <= 8'h00;

            // Set FF (no enable)
            #100 // 100 ns delay
            t_clock <= 1'b1;
            t_reset <= 1'b0;
            t_enable <= 1'b0;
            t_d <= 8'hFF;
            #100 // 100 ns delay
            t_clock <= 1'b0;
            
            // Set FF (with enable)
            #100 // 100 ns delay
            t_clock <= 1'b1;
            t_reset <= 1'b0;
            t_enable <= 1'b1;
            t_d <= 8'hFF;

            // Reset
            #100 // 100 ns delay
            t_clock <= 1'b0;
            t_reset <= 1'b1;
            t_enable <= 1'b0;
            t_d <= 8'hFF;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule