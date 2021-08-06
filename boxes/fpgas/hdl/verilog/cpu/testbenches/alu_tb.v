// Testbench for ALU
module alu_tb;

    // Declarations
    reg [31:0] t_src_A;
    reg [31:0] t_src_B;
    reg [2:0] t_ALU_control;
    wire [31:0] t_ALU_result;
    wire t_zero;

    // Create instance of alu module
    alu test_alu(t_src_A, t_src_B, t_ALU_control, t_ALU_result, t_zero);

    // Test
    initial
        begin
            $dumpfile("bin/alu_tb.vcd");
            $dumpvars(0, alu_tb);
            $monitor(t_src_A, t_src_B, t_ALU_control, t_ALU_result, t_zero);
            
            // Initialize
            #100; // 100 ns delay
            t_src_A = 32'h00000000;
            t_src_B = 32'h00000000;
            t_ALU_control = 3'b000;

            // A + B
            #100; // 100 ns delay
            t_src_A = 32'h00000001;
            t_src_B = 32'h00000001;
            t_ALU_control = 3'b000;

            // A - B
            #100; // 100 ns delay
            t_src_A = 32'h00000001;
            t_src_B = 32'h00000001;
            t_ALU_control = 3'b001;
        end

endmodule