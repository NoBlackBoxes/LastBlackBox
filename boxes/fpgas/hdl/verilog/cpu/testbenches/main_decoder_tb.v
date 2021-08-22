// Testbench for Main Decoder
module main_decoder_tb;

    // Declarations
    reg [6:0] t_opcode;
    reg [6:0] t_funct3;
    wire t_reg_write;
    wire t_ALU_select;
    wire [3:0] t_memory_control;
    wire [2:0] t_result_select;
    wire t_branch;
    wire [1:0] t_ALU_op;
    wire t_jump;

    // Create instance of main_decoder module
    main_decoder test_main_decoder(t_opcode, t_funct3, t_reg_write, t_ALU_select, t_memory_control, t_result_select, t_branch, t_ALU_op, t_jump);

    // Test
    initial
        begin
            $dumpfile("bin/main_decoder_tb.vcd");
            $dumpvars(0, main_decoder_tb);
            $monitor(t_opcode, t_funct3, t_reg_write, t_ALU_select, t_memory_control, t_result_select, t_branch, t_ALU_op, t_jump);
            
            // Initialize
            t_opcode <= 7'b0000000;
            #100; // 100 ns delay

            // Test lw
            t_opcode <= 7'b0000011;
            #100; // 100 ns delay

            // Test sw
            t_opcode <= 7'b0100011;
            #100; // 100 ns delay

            // Test R–type
            t_opcode <= 7'b0110011;
            #100; // 100 ns delay

            // Test beq
            t_opcode <= 7'b1100011;
            #100; // 100 ns delay

            // Test I–type ALU
            t_opcode <= 7'b0010011;
            #100; // 100 ns delay

            // Test jal
            t_opcode <= 7'b1101111;
            #100; // 100 ns delay

            // Test lui
            t_opcode <= 7'b0110111;
            #100; // 100 ns delay

            // Test ???
            t_opcode <= 7'b1111111;
            #100; // 100 ns delay
        end

endmodule