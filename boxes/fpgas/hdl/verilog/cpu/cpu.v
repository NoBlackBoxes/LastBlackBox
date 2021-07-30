// CPU (RV32I)
module cpu(clock, reset, instruction, read_data, mem_write, PC, ALU_result, write_data);
    
    // Declarations
    input clock;
    input reset;
    input [31:0] instruction;
    input [31:0] read_data;
    output mem_write;
    output [31:0] PC;
    output [31:0] ALU_result;
    output [31:0] write_data;
    
    // Intermediates
    wire PC_src; 
    wire ALU_src; 
    wire reg_write;
    wire jump;
    wire zero;
    wire [1:0] result_src;
    wire [1:0] imm_src;
    wire [2:0] ALU_control;

    // Sub-module: Controller
    controller controller
    (
        instruction[6:0],       // (input) Field: opcode
        instruction[14:12],     // (input) Field: funct3
        instruction[30],        // (input) funct7b5
        zero,                   // (input) zero
        result_src,             // (output) result_src
        mem_write,              // (output) mem_write
        PC_src,                 // (output) PC_src
        ALU_src,                // (output) ALU_Src
        reg_write,              // (output) reg_write
        jump,                   // (output) jump
        imm_src,                // (output) imm_src
        ALU_control             // (output) ALU_control
    );

    // Sub-module: Datapath
    datapath datapath
    (
        clock,                  // (input) clock
        reset,                  // (input) reset
        result_src,             // (input) result_src
        PC_src,                 // (input) PC_src
        ALU_src,                // (input) ALU_src
        reg_write,              // (input) reg_write
        imm_src,                // (input) imm_src
        ALU_control,            // (input) ALU_control
        instruction,            // (input) instruction
        read_data,              // (input) read_data
        zero,                   // (output) zero
        PC,                     // (output) PC
        ALU_result,             // (output) ALU_result
        write_data              // (output) write_data
    );

endmodule