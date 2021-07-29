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
    ?wire ALU_src; 
    ?wire reg_write;
    ?wire jump;
    ?wire zero;
    ?wire [1:0] ResultSrc, ImmSrc;
    ?wire [2:0] ALUControl;

    // Controller Sub-module
    controller controller(Instr[6:0], Instr[14:12], Instr[30], Zero,ResultSrc, MemWrite, PCSrc, ALUSrc, RegWrite, Jump, ImmSrc, ALUControl);

    // Datapath Sub-module
    datapath datapath(clk, reset, ResultSrc, PCSrc,ALUSrc,  RegWrite,ImmSrc,  ALUControl,Zero, PC, Instr,ALUResult, WriteData, ReadData);

endmodule