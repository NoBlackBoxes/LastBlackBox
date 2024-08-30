// Testbench for Mux4
module mux8_tb;

    // Declarations
    reg [3:0] t_d0;
    reg [3:0] t_d1;
    reg [3:0] t_d2;
    reg [3:0] t_d3;
    reg [3:0] t_d4;
    reg [3:0] t_d5;
    reg [3:0] t_d6;
    reg [3:0] t_d7;
    reg [2:0] t_s;
    wire [3:0] t_y;

    // Create instance of mux8 module
    defparam test_mux8.WIDTH = 4;
    mux8 test_mux8(t_d0, t_d1, t_d2, t_d3, t_d4, t_d5, t_d6, t_d7, t_s, t_y);

    // Test
    initial
        begin
            $dumpfile("bin/mux8_tb.vcd");
            $dumpvars(0, mux8_tb);
            $monitor(t_d0, t_d1, t_d2, t_d3, t_d4, t_d5, t_d6, t_d7, t_s, t_y);

            // Initial
            t_d0 <= 4'h0;
            t_d1 <= 4'h2;
            t_d2 <= 4'h4;
            t_d3 <= 4'h6;
            t_d4 <= 4'h8;
            t_d5 <= 4'hA;
            t_d6 <= 4'hC;
            t_d7 <= 4'hE;
            t_s <= 3'b00;

            // Select 0
            #100 // 100 ns delay
            t_s <= 3'b000;
            
            // Select 1
            #100 // 100 ns delay
            t_s <= 3'b001;

            // Select 2
            #100 // 100 ns delay
            t_s <= 3'b010;

            // Select 3
            #100 // 100 ns delay
            t_s <= 3'b011;

            // Select 4
            #100 // 100 ns delay
            t_s <= 3'b100;

            // Select 5
            #100 // 100 ns delay
            t_s <= 3'b101;

            // Select 6
            #100 // 100 ns delay
            t_s <= 3'b110;

            // Select 7
            #100 // 100 ns delay
            t_s <= 3'b111;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule