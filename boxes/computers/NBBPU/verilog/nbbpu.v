// NBBPU
// -----------------------------------------
// This is the top module of the NBBPU (CPU). It receives a clock, reset signal, and 16-bit instruction. It also receives "data in" from RAM.
// It outputs control signals for memory (memory_control), the value of the progam counter (PC), a data address, and "data out" for RAM.
// Not all of the outputs (or inputs) are used for every instruction, but the connections always exists and the logic decides what gets used.
//
// The nbbpu module invokes two sub-modules (controller and datapath)
// -----------------------------------------
module nbbpu(clock, reset, instruction, data_in, data_write, data_address, data_out, PC);
    
    // Declarations
    input clock;
    input reset;
    input [15:0] instruction;
    input [15:0] data_in;
    output data_write;
    output [15:0] data_address;
    output [15:0] data_out;
    output [15:0] PC;

    // Intermediates
    wire reg_write;
    wire reg_set;
    wire data_write;
    wire PC_select;

    // Sub-module: Controller
    controller controller
    (
        instruction[15:12],   // (input) Opcode
        reg_write,            // (output) Register write enable
        reg_set,              // (output) Register set enable
        data_write,           // (output) Data write enable
        PC_select             // (output) PC select signal
    );

    // Sub-module: Datapath
    datapath datapath
    (
        clock,                // (input) clock
        reset,                // (input) reset
        instruction,          // (input) instruction
        reg_write,            // (input) reg_write
        reg_set,              // (input) reg_set
        PC_select,            // (input) PC select signal
        data_in,              // (input) data_in
        data_out,             // (output) data_out
        PC                    // (output) PC
    );

endmodule