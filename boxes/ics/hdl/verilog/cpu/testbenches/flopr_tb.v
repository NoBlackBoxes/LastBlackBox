// Testbench for Flopr
module flopr_tb;

    // Declarations
    reg t_clock;
    reg t_reset;
    reg [7:0] t_d;
    wire [7:0] t_q;

    // Create instance of flopr module
    defparam test_flopr.WIDTH = 8;
    flopr test_flopr(t_clock, t_reset, t_d, t_q);

    // Test
    initial
        begin
            $dumpfile("bin/flopr_tb.vcd");
            $dumpvars(0, flopr_tb);
            $monitor(t_clock, t_reset, t_d, t_q);

            // Initial
            t_clock <= 1'b0;
            t_reset <= 1'b0;
            t_d <= 8'h00;

            // Set FF
            #100 // 100 ns delay
            t_clock <= 1'b1;
            t_reset <= 1'b0;
            t_d <= 8'hFF;
            
            // Reset
            #100 // 100 ns delay
            t_clock <= 1'b0;
            t_reset <= 1'b1;
            t_d <= 8'hFF;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule