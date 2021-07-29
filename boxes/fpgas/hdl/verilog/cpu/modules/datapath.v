// Datapath (RV32I)
module datapath(clock, reset,

    // Declarations
    input clock, 
    input reset,
    input [1:0] ResultSrc,
    input PCSrc, 
    input ALUSrc, 
    input RegWrite,
    input [1:0] ImmSrc,
    input [2:0] ALUControl,
    output Zero,
    output logic [31:0] PC,
    input [31:0]  Instr,
    output [31:0] ALUResult, 
    output [31:0] WriteData,
    input [31:0]  ReadData);
    
    // Intermediates
    logic [31:0] PCNext, PCPlus4, PCTarget;   
    logic [31:0] ImmExt;   
    logic [31:0] SrcA, SrcB;   
    logic [31:0] Result;   
    
    // Logic (next PC)   
    flopr #(32) pcreg(clk, reset, PCNext, PC);
    adder pcadd4(PC, 32'd4, PCPlus4);   
    adder pcaddbranch(PC, ImmExt, PCTarget);   
    mux2 #(32)pcmux(PCPlus4, PCTarget, PCSrc, PCNext);   
    
    // Logic (register file)
    regfile regfile(clk, RegWrite, Instr[19:15], Instr[24:20],Instr[11:7], Result, SrcA, WriteData);
    extend extend(Instr[31:7], ImmSrc, ImmExt);   
    
    // Logic (ALU)
    mux2 #(32) srcbmux(WriteData, ImmExt, ALUSrc, SrcB);
    alu alu(SrcA, SrcB, ALUControl, ALUResult, Zero);
    mux3 #(32) resultmux(ALUResult, ReadData, PCPlus4,ResultSrc, Result);

endmodule