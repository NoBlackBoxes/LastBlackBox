// Testbench for Mux4
module mux4_tb;

    // Declarations
    reg [3:0] t_d0;
    reg [3:0] t_d1;
    reg [3:0] t_d2;
    reg [3:0] t_d3;
    reg [1:0] t_s;
    wire [3:0] t_y;

    // Create instance of mux4 module
    defparam test_mux4.WIDTH = 4;
    mux4 test_mux4(t_d0, t_d1, t_d2, t_d3, t_s, t_y);

    // Test
    initial
        begin
            $dumpfile("bin/mux4_tb.vcd");
            $dumpvars(0, mux4_tb);
            $monitor(t_d0, t_d1, t_d2, t_d3, t_s, t_y);

            // Initial
            t_d0 <= 4'h0;
            t_d1 <= 4'h0;
            t_d2 <= 4'h0;
            t_d3 <= 4'h0;
            t_s = 2'b00;

            // Select 0
            #100 // 100 ns delay
            t_d0 <= 4'h0;
            t_d1 <= 4'h1;
            t_d2 <= 4'h5;
            t_d3 <= 4'hF;
            t_s = 2'b00;
            
            // Select 1
            #100 // 100 ns delay
            t_d0 <= 4'h0;
            t_d1 <= 4'h1;
            t_d2 <= 4'h5;
            t_d3 <= 4'hF;
            t_s = 2'b01;

            // Select 2
            #100 // 100 ns delay
            t_d0 <= 4'h0;
            t_d1 <= 4'h1;
            t_d2 <= 4'h5;
            t_d3 <= 4'hF;
            t_s = 2'b10;

            // Select 3
            #100 // 100 ns delay
            t_d0 <= 4'h0;
            t_d1 <= 4'h1;
            t_d2 <= 4'h5;
            t_d3 <= 4'hF;
            t_s = 2'b11;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule