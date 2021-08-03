// Testbench for Mux3
module mux3_tb;

    // Declarations
    reg [3:0] t_d0;
    reg [3:0] t_d1;
    reg [3:0] t_d2;
    reg [1:0] t_s;
    wire [3:0] t_y;

    // Create instance of mux3 module
    defparam test_mux3.WIDTH = 4;
    mux3 test_mux3(t_d0, t_d1, t_d2, t_s, t_y);

    // Test
    initial
        begin
            $dumpfile("bin/mux3_tb.vcd");
            $dumpvars(0, mux3_tb);
            $monitor(t_d0, t_d1, t_d2, t_s, t_y);

            // Initial
            t_d0 <= 4'h0;
            t_d1 <= 4'h0;
            t_d2 <= 4'h0;
            t_s = 2'b00;

            // Select 0
            #100 // 100 ns delay
            t_d0 <= 4'h0;
            t_d1 <= 4'h1;
            t_d2 <= 4'hF;
            t_s = 2'b00;
            
            // Select 1
            #100 // 100 ns delay
            t_d0 <= 4'h0;
            t_d1 <= 4'h1;
            t_d2 <= 4'hF;
            t_s = 2'b01;

            // Select 2
            #100 // 100 ns delay
            t_d0 <= 4'h0;
            t_d1 <= 4'h1;
            t_d2 <= 4'hF;
            t_s = 2'b10;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule