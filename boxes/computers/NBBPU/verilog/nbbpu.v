// NBBPU
// -----------------------------------------
// This is the top module of the NBBPU (CPU). It receives a clock, reset signal, and 16-bit instruction. It also receives "data in" from RAM.
// It outputs control signals for memory (memory_control), the value of the progam counter (PC), and "data out" for RAM.
// Not all of the outputs (or inputs) are used for every instruction, but the connections always exists and the logic decides what gets used.
//
// It also generates some internal signals (intermediates) that are used by sub-modules. PC_select and ALU_select do something.
// Reg_write also does something. Jump specifies that a PC jump will occur. Zero indictaes that the ALU result was zero. Result select determines
// which result is selected and ALU_control...controls the ALU.
//
// The nbbpu module invokes two sub-modules (controller and datapath)
// -----------------------------------------
module nbbpu(clock, reset, instruction, data_in, PC, memory_control, data_out);
    
    // Declarations
    input clock;
    input reset;
    input [15:0] instruction;
    input [15:0] data_in;
    output [15:0] PC;
    output [3:0] memory_control;
    output [15:0] data_out;
    
    // Intermediates
    wire [15:0] PC_next;
    wire [1:0] PC_select;
    wire ALU_select; 
    wire reg_write;
    wire jump;
    wire zero;
    wire [2:0] result_select;
    wire [3:0] ALU_control;
            
//    // Sub-module: Controller
//    controller controller
//    (
//        instruction[3:0],       // (input) Field: opcode
//        instruction[7:4],       // (input) Field: x
//        instruction[11:8],      // (input) Field: y
//        instruction[15:12],     // (input) Field: z
//        zero,                   // (input) zero
//        result_select,          // (output) result_select
//        memory_control,         // (output) memory_control
//        PC_select,              // (output) PC_src
//        ALU_select,             // (output) ALU_Src
//        reg_write,              // (output) reg_write
//        jump,                   // (output) jump
//        ALU_control             // (output) ALU_control
//    );
//
//    // Sub-module: Datapath
//    datapath datapath
//    (
//        clock,                  // (input) clock
//        reset,                  // (input) reset
//        result_select,          // (input) result_select
//        PC_select,              // (input) PC_select
//        ALU_select,             // (input) ALU_select
//        reg_write,              // (input) reg_write
//        ALU_control,            // (input) ALU_control
//        instruction,            // (input) instruction
//        data_in,              // (input) data_in
//        zero,                   // (output) zero
//        PC,                     // (output) PC
//        ALU_result,             // (output) ALU_result
//        data_out              // (output) data_out
//    );

    flopr #(16) pc_reg(clock, reset, PC_next, PC);
    adder pc_add2(PC, 16'd2, PC_next);

endmodule