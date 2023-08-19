// NBBSOC
// --------------------------------------------------------
// This is the top module of the NBBSOC (System-on-a-chip). 
// --------------------------------------------------------
module nbbsoc(clock, reset, blink);
    
    // Declarations
    input clock;
    input reset;
    output blink;

    // Intermediates
    wire select;
    wire write_enable;
    wire [15:0] instruction;
    wire [15:0] read_data;
    wire [15:0] address;
    wire [15:0] write_data;
    wire [15:0] PC;

    // Assignments
    assign select = 1'b1;

    // CPU module
    nbbpu nbbpu(clock, reset, instruction, read_data, write_enable, address, write_data, PC, debug);
    
    // Create Instruction and Data Memory modules
    rom rom(clock, select, PC, instruction);
    ram ram(clock, select, write_enable, address, write_data, read_data);
    
    // Assign Debug signals
    assign blink = debug;

endmodule