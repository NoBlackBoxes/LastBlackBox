// Controller (RV32I)
module controller(opcode, funct3, funct7b5, zero, result_src, mem_write, PC_src, ALU_src, reg_write, jump, imm_src, ALU_control);

    // Declarations
    input [6:0] opcode,
    input [2:0] funct3,       
    input funct7b5,                     
    input zero,                     
    output [1:0] result_src,                     
    output mem_write,                     
    output PC_src;
    output ALU_src;                     
    output reg_write;
    output jump;                     
    output [1:0] imm_src;                
    output [2:0] ALU_control;
    
    // Intermediates
    wire [1:0] ALU_op;
    wire branch;
    
    // Main Decoder Sub-module
    decoder_main decoder_main
    (
        opcode,         // (input) opcode
        result_src,     // (output)
        mem_write,      // (output)
        branch,         // (output)
        ALU_src,        // (output)
        reg_write,      // (output)
        jump,           // (output)
        imm_src,        // (output)
        ALU_op          // (output)
    );
    
    // ALU Decoder Sub-module
    decoder_ALU decoder_ALU
    (
        opcode[5],      // (input) opcode_b5
        funct3,         // (input)
        funct7b5,       // (input)
        ALU_op,         // (input)
        ALU_control     // (output)
    );

    // Program Counter update (?)   
    assign PC_src = branch & zero | jump;
    
endmodule