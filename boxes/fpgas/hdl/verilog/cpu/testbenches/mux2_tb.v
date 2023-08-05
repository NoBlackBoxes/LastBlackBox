// Testbench for Mux2
module mux2_tb;

    // Declarations
    reg [3:0] t_d0;
    reg [3:0] t_d1;
    reg t_s;
    wire [3:0] t_y;

    // Create instance of mux2 module
    defparam test_mux2.WIDTH = 4;
    mux2 test_mux2(t_d0, t_d1, t_s, t_y);

    // Test
    initial
        begin
            $dumpfile("bin/mux2_tb.vcd");
            $dumpvars(0, mux2_tb);
            $monitor(t_d0, t_d1, t_s, t_y);

            // Initial
            t_d0 <= 4'h0;
            t_d1 <= 4'h0;
            t_s = 1'b0;

            // Select 0
            #100 // 100 ns delay
            t_d0 <= 4'h1;
            t_d1 <= 4'hF;
            t_s = 1'b0;
            
            // Select 1
            #100 // 100 ns delay
            t_d0 <= 4'h1;
            t_d1 <= 4'hF;
            t_s = 1'b1;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule