// NBBPU
// -----------------------------------------
// This is the top module of the NBBPU (CPU). It receives a clock, reset signal, and 16-bit instruction.
// It also receives "read_data" coming from RAM.
// It outputs a control signal for memory ("write_enable"), the address to write to ("wrtie_address"),
// and the value to write ("write_data").
// It also updates the the value of the program counter (PC) to retrieve the next instruction.
//
// Not all of the outputs (or inputs) are used for every instruction,
// but the connections always exist and the logic decides what gets used and what is ignored.
//
// The nbbpu module invokes two sub-modules (controller and datapath)
// -----------------------------------------
module nbbpu(clock, reset, instruction, read_data, write_enable, address, write_data, PC);
    
    // Declarations
    input clock;
    input reset;
    input [15:0] instruction;
    input [15:0] read_data;
    output write_enable;
    output [15:0] address;
    output [15:0] write_data;
    output [15:0] PC;

    // Intermediates
    wire reg_write;
    wire write_enable;
    wire jump_PC;
    wire branch_PC;

    // Sub-module: Controller
    controller controller
    (
        instruction[15:12],     // (input) Opcode
        reg_write,              // (output) Register write enable
        write_enable,           // (output) Data write enable
        jump_PC,                // (output) jump PC signal
        branch_PC               // (output) branch PC signal
    );

    // Sub-module: Datapath
    datapath datapath
    (
        clock,                  // (input) clock
        reset,                  // (input) reset
        instruction,            // (input) instruction
        reg_write,              // (input) reg_write
        jump_PC,                // (input) jump_PC
        branch_PC,              // (input) branch_PC
        read_data,              // (input) read_data
        address,                // (output) address
        write_data,             // (output) write_data
        PC                      // (output) PC
    );

endmodule