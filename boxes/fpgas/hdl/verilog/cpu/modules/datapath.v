// Datapath (RV32I)
module datapath(clock, reset, result_src, PC_src, ALU_src, reg_write, imm_src, ALU_control, instruction, read_data, zero, PC, ALU_result, write_data);

    // Declarations
    input clock; 
    input reset;
    input [1:0] result_src;
    input PC_src;
    input ALU_src;
    input reg_write;
    input [1:0] imm_src;
    input [2:0] ALU_control;
    input [31:0] instruction;
    input [31:0] read_data;
    output zero;
    output [31:0] PC;
    output [31:0] ALU_result; 
    output [31:0] write_data;
    
    // Intermediates
    wire [31:0] PC_next
    wire [31:0] PC_plus4;
    wire [31:0] PC_target;   
    wire [31:0] imm_ext;   
    wire [31:0] srcA;
    wire [31:0] srcB;   
    wire [31:0] result;   
    
    // Logic (next PC)   
    flopr #(32) pcreg(clock, reset, PC_next, PC);
    adder pcadd4(PC, 32'd4, PC_plus4);   
    adder pcaddbranch(PC, imm_ext, PC_target);   
    mux2 #(32) pcmux(PC_plus4, PC_target, PC_src, PC_next);   
    
    // Logic (register file)
    regfile regfile(clock, RegWrite, Instr[19:15], Instr[24:20],Instr[11:7], Result, SrcA, WriteData);
    extend extend(Instr[31:7], ImmSrc, ImmExt);   
    
    // Logic (ALU)
    mux2 #(32) srcbmux(WriteData, ImmExt, ALUSrc, SrcB);
    alu alu(SrcA, SrcB, ALUControl, ALUResult, Zero);
    mux3 #(32) resultmux(ALUResult, ReadData, PCPlus4,ResultSrc, Result);

endmodule