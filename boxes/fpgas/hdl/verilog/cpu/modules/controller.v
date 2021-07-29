// Controller (RV32I)
module controller(opcode, funct3, funct7b5, zero, result_src,mem_write);

    // Declarations
    input logic [6:0] opcode,
    input logic [2:0] funct3,       
    input logic funct7b5,                     
    input logic zero,                     
    output logic [1:0] ResultSrc,                     
    output logic          mem_write,                     
    output logic          PCSrc, ALUSrc,                     
    output logic          RegWrite, Jump,                     
    output logic [1:0] ImmSrc,                     
    output logic [2:0] ALUControl);
    
    // Intermediates
    logic [1:0] ALUOp;
    logic Branch;
    
    // Main Decoder Sub-module
    decoder_main decoder_main(op, ResultSrc, MemWrite, Branch, ALUSrc, RegWrite, Jump, ImmSrc, ALUOp);
    
    // ALU Decoder Sub-module
    decoder_ALU  decoder_ALU(op[5], funct3, funct7b5, ALUOp, ALUControl);

    // Program Counter update (?)   
    assign PCSrc = Branch & Zero | Jump;
    
endmodule