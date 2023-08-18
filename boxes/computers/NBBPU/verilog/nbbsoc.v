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
    wire write_enable;
    wire [15:0] instruction;
    wire [15:0] read_data;
    wire [15:0] address;
    wire [15:0] write_data;
    wire [15:0] PC;

    // CPU module
    nbbpu nbbpu(clock, reset, instruction, read_data, write_enable, address, write_data, PC);
    
    // Create Instruction and Data Memory modules
    rom rom(PC, instruction);
    ram ram(clock, write_enable, address, write_data, read_data);

    // Assign Debug signals
    assign blink = write_enable;

endmodule