// NBBSOC
// --------------------------------------------------------
// This is the top module of the NBBSOC (System-on-a-chip). 
// --------------------------------------------------------
module nbbsoc(clock, reset, RGB);
    
    // Declarations
    input clock;
    input reset;
    output [2:0] RGB;

    // Intermediates
    wire select;
    wire [15:0] instruction;
    wire [15:0] read_data;
    wire instruction_enable;
    wire read_enable;
    wire write_enable;
    wire [15:0] address;
    wire [15:0] write_data;
    wire [15:0] PC;
    wire [2:0] debug_RGB;

    // Assignments
    assign select = 1'b1;

    // CPU module
    nbbpu nbbpu(
                clock, 
                reset, 
                instruction, 
                read_data, 
                instruction_enable, 
                read_enable, 
                write_enable, 
                address, 
                write_data, 
                PC, 
                debug_RGB);
    
    // Create Instruction and Data Memory modules
    rom rom(clock, select, instruction_enable, PC, instruction);
    ram ram(clock, select, read_enable, write_enable, address, write_data, read_data);
    
    // Assign Debug signals
    assign RGB = debug_RGB;

endmodule