// Testbench for Controller
module controller_tb;

    // Declarations
    reg [6:0] t_opcode;
    reg [2:0] t_funct3;
    reg t_funct7b5;
    reg t_zero;
    wire [2:0] t_result_select;
    wire t_mem_write;
    wire t_PC_select;
    wire t_ALU_select;
    wire t_reg_write;
    wire t_jump;
    wire [2:0] t_ALU_control;

    // Create instance of controller module
    controller test_controller(t_opcode, t_funct3, t_funct7b5, t_zero, t_result_select, t_mem_write, t_PC_select, t_ALU_select, t_reg_write, t_jump, t_ALU_control);

    // Test
    initial
        begin
            $dumpfile("bin/controller_tb.vcd");
            $dumpvars(0, controller_tb);
            $monitor(t_opcode, t_funct3, t_funct7b5, t_zero, t_result_select, t_mem_write, t_PC_select, t_ALU_select, t_reg_write, t_jump, t_ALU_control);
            
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

            // Test ???
            t_opcode <= 7'b1111111;
            #100; // 100 ns delay
        end

endmodule