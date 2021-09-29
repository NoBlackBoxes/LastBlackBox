// Testbench for ALU Decoder
module alu_decoder_tb;

    // Declarations
    reg t_opcode_b5;
    reg [2:0] t_funct3;
    reg t_funct7b5;
    reg [1:0] t_ALU_op;
    wire [2:0] t_ALU_control;

    // Create instance of alu_decoder module
    alu_decoder test_alu_decoder(t_opcode_b5, t_funct3, t_funct7b5, t_ALU_op, t_ALU_control);

    // Test
    initial
        begin
            $dumpfile("bin/alu_decoder_tb.vcd");
            $dumpvars(0, alu_decoder_tb);
            $monitor(t_opcode_b5, t_funct3, t_funct7b5, t_ALU_op, t_ALU_control);
            
            // Initialize
            t_opcode_b5 <= 1'b0;
            t_funct3 <= 3'b000;
            t_funct7b5 <= 1'b0;
            t_ALU_op <= 2'b00;
            #100; // 100 ns delay

            // Test addition
            t_ALU_op <= 2'b00;
            #100; // 100 ns delay

            // Test subtraction
            t_ALU_op <= 2'b01;
            #100; // 100 ns delay

            // Test R-type subtraction
            t_ALU_op <= 2'b11;
            t_funct3 <= 3'b000;
            t_opcode_b5 <= 1'b1;
            t_funct7b5 <= 1'b1;
            #100; // 100 ns delay

            // Test R-type addition
            t_ALU_op <= 2'b11;
            t_funct3 <= 3'b000;
            t_opcode_b5 <= 1'b0;
            t_funct7b5 <= 1'b1;
            #100; // 100 ns delay

            // Test slt, slti
            t_ALU_op <= 2'b11;
            t_funct3 <= 3'b010;
            #100; // 100 ns delay

            // Test or, ori
            t_ALU_op <= 2'b11;
            t_funct3 <= 3'b110;
            #100; // 100 ns delay

            // Test and, andi
            t_ALU_op <= 2'b11;
            t_funct3 <= 3'b111;
            #100; // 100 ns delay

            // Test ???
            t_ALU_op <= 2'b11;
            t_funct3 <= 3'b011;
            #100; // 100 ns delay
        end

endmodule