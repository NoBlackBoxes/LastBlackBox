// Controller (NBBPU)
// -----------------------------------------
// This is the "controller" sub-module for the NBBPU. It is responsible for decoding the opcode and generating the required
// control signals. 
// -----------------------------------------
module controller(opcode, reg_write, write_enable, jump_PC, branch_PC);

    // Declarations
    input [3:0] opcode;
    output reg_write;
    output write_enable;
    output jump_PC;
    output branch_PC;

    // Intermediates
    reg [4:0] controls;
    assign {reg_write, write_enable, jump_PC, branch_PC} = controls;
   
    // Logic
    always @*
        case(opcode)      
            4'b0000: // ADD
                controls = 4'b1000;
            4'b0001: // SUB
                controls = 4'b1000;
            4'b0010: // AND 
                controls = 4'b1000;
            4'b0011: // IOR
                controls = 4'b1000;
            4'b0100: // XOR
                controls = 4'b1000;
            4'b0101: // SHR
                controls = 4'b1000;
            4'b0110: // SHL
                controls = 4'b1000;
            4'b0111: // CMP
                controls = 4'b1000;
            4'b1000: // JMP
                controls = 4'b1010;
            4'b1001: // BRZ
                controls = 4'b0001;
            4'b1010: // BRN
                controls = 4'b0001;
            4'b1011: // RES
                controls = 4'b1000;
            4'b1100: // LOD
                controls = 4'b1000;
            4'b1101: // STR
                controls = 4'b0100;
            4'b1110: // SEL
                controls = 4'b1000;
            4'b1111: // SEU
                controls = 4'b1000;
            default: // ???
                controls = 4'b0000;
        endcase

endmodule