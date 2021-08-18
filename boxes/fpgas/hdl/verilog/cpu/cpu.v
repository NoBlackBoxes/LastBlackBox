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
    wire PC_select; 
    wire ALU_select; 
    wire reg_write;
    wire jump;
    wire zero;
    wire [1:0] result_select;
    wire [2:0] ALU_control;

    // Sub-module: Controller
    controller controller
    (
        instruction[6:0],       // (input) Field: opcode
        instruction[14:12],     // (input) Field: funct3
        instruction[30],        // (input) funct7b5
        zero,                   // (input) zero
        result_select,          // (output) result_src
        mem_write,              // (output) mem_write
        PC_select,              // (output) PC_src
        ALU_select,             // (output) ALU_Src
        reg_write,              // (output) reg_write
        jump,                   // (output) jump
        ALU_control             // (output) ALU_control
    );

    // Sub-module: Datapath
    datapath datapath
    (
        clock,                  // (input) clock
        reset,                  // (input) reset
        result_select,          // (input) result_select
        PC_select,              // (input) PC_select
        ALU_select,             // (input) ALU_select
        reg_write,              // (input) reg_write
        ALU_control,            // (input) ALU_control
        instruction,            // (input) instruction
        read_data,              // (input) read_data
        zero,                   // (output) zero
        PC,                     // (output) PC
        ALU_result,             // (output) ALU_result
        write_data              // (output) write_data
    );

endmodule