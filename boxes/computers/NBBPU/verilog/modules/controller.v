// Controller (NBBPU)
// -----------------------------------------
// This is the "controller" sub-module for the NBBPU. It is responsible for decoding the opcode and generating the required
// control signals. 
// -----------------------------------------
module controller(opcode, reg_write, reg_set, write_enable, jump_PC, branch_PC);

    // Declarations
    input [3:0] opcode;
    output reg_write;
    output reg_set;
    output write_enable;
    output jump_PC;
    output branch_PC;

    // Intermediates
    reg [4:0] controls;
    assign {reg_write, reg_set, write_enable, jump_PC, branch_PC} = controls;
   
    // Logic
    always @*
        case(opcode)      
            4'b0000: // ADD
                controls = 5'b10000;
            4'b0001: // SUB
                controls = 5'b10000;
            4'b0010: // AND 
                controls = 5'b10000;
            4'b0011: // IOR
                controls = 5'b10000;
            4'b0100: // XOR
                controls = 5'b10000;
            4'b0101: // SHR
                controls = 5'b10000;
            4'b0110: // SHL
                controls = 5'b10000;
            4'b0111: // CMP
                controls = 5'b10000;
            4'b1000: // JMP
                controls = 5'b10010;
            4'b1001: // BRZ
                controls = 5'b00001;
            4'b1010: // BRN
                controls = 5'b00001;
            4'b1011: // RES
                controls = 5'b10000;
            4'b1100: // LOD
                controls = 5'b10000;
            4'b1101: // STR
                controls = 5'b00100;
            4'b1110: // SEL
                controls = 5'b11000;
            4'b1111: // SEU
                controls = 5'b11000;
            default: // ???
                controls = 5'b00000;
        endcase

endmodule