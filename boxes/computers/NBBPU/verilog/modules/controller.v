// Controller (NBBPU)
// -----------------------------------------
// This is the "controller" sub-module for the NBBPU. It is responsible for decoding the opcode and generating the required
// control signals. 
// -----------------------------------------
module controller(opcode, reg_write_lower, reg_write_upper, reg_set, write_enable, PC_select);

    // Declarations
    input [3:0] opcode;
    output reg_write_lower;
    output reg_write_upper;
    output reg_set;
    output write_enable;
    output PC_select;

    // Intermediates
    reg [5:0] controls;
    assign {reg_write_lower, reg_write_upper, reg_set, write_enable, PC_select} = controls;
   
    // Logic
    always @*
        case(opcode)      
            4'b0000: // ADD
                controls = 5'b11000;
            4'b0001: // SUB
                controls = 5'b11000;
            4'b0010: // AND 
                controls = 5'b11000;
            4'b0011: // IOR
                controls = 5'b11000;
            4'b0100: // XOR
                controls = 5'b11000;
            4'b0101: // SHR
                controls = 5'b11000;
            4'b0110: // SHL
                controls = 5'b11000;
            4'b0111: // CMP
                controls = 5'b11000;
            5'b11000: // JMP
                controls = 5'b11000;
            5'b11001: // BRE
                controls = 5'b11000;
            5'b11010: // BRN
                controls = 5'b11000;
            5'b11011: // RES
                controls = 5'b11000;
            4'b1100: // LOD
                controls = 5'b11000;
            4'b1101: // STR
                controls = 5'b11010;
            4'b1110: // SEL
                controls = 5'b10100;
            4'b1111: // SEU
                controls = 5'b01100;
            default: // ???
                controls = 5'b11000;
        endcase

//    // Program Counter update for branch instructions
//    always @*
//        if(jump)
//            case(opcode[3])
//                1'b0: PC_select = 5'b11000;            // jalr
//                1'b1: PC_select = 2'b01;            // jal
//                default: PC_select = 2'b01;
//            endcase
//        else
//            case(funct3)
//                3'b000: PC_select = branch & zero;   // beq
//                3'b001: PC_select = branch & ~zero;  // bne
//                3'b100: PC_select = branch & ~zero;  // blt
//                3'b101: PC_select = branch & zero;   // bge
//                3'b110: PC_select = branch & ~zero;  // bltu
//                3'b111: PC_select = branch & zero;   // bgeu
//                default: PC_select = 5'b11000;
//            endcase           
//        
//        // Compute memory offset for misaligned read/writes

endmodule