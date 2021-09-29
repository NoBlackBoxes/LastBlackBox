// Testbench for Adder
module adder_tb;

    // Declarations
    reg [31:0] t_a;
    reg [31:0] t_b;
    wire [31:0] t_sum;

    // Create instance of adder module
    adder test_adder(t_a, t_b, t_sum);

    // Test
    initial
        begin
            $dumpfile("bin/adder_tb.vcd");
            $dumpvars(0, adder_tb);
            $monitor(t_a, t_b, t_sum);

            // 0 + 0
            #100 // 100 ns delay
            t_a <= 32'h00000000;
            t_b <= 32'h00000000;
            
            // 1 + 1
            #100 // 100 ns delay
            t_a <= 32'h00000001;
            t_b <= 32'h00000001;

            // 16 + 1
            #100 // 100 ns delay
            t_a <= 32'h000000FF;
            t_b <= 32'h00000001;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule