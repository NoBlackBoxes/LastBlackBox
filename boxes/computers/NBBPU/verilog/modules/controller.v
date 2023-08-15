// Controller (NBBPU)
// -----------------------------------------
// This is the "controller" sub-module for the NBBPU. It is responsible for decoding the opcode and generating the required
// control signals. 
// -----------------------------------------
module controller(opcode, reg_write, data_write, pc_select);

    // Declarations
    input [3:0] opcode;
    output reg_write;
    output data_write;
    output pc_select;
    
    // Intermediates
    reg [3:0] controls;
    assign {reg_write, data_write, pc_select} = controls;
   
    // Logic
    always @*
        case(opcode)      
            4'b0000: // ADD
                controls = 3'b100;
            4'b0001: // SUB
                controls = 3'b100;
            4'b0010: // AND 
                controls = 3'b100;
            4'b0011: // IOR
                controls = 3'b100;
            4'b0100: // XOR
                controls = 3'b100;
            4'b0101: // SHR
                controls = 3'b100;
            4'b0110: // SHL
                controls = 3'b100;
            4'b0111: // CMP
                controls = 3'b100;
            4'b1000: // JMP
                controls = 3'b100;
            4'b1001: // BRE
                controls = 3'b100;
            4'b1010: // BRN
                controls = 3'b100;
            4'b1011: // RES
                controls = 3'b100;
            4'b1100: // LOD
                controls = 3'b100;
            4'b1101: // STR
                controls = 2'b11;
            4'b1110: // SEL
                controls = 3'b100;
            4'b1111: // SEU
                controls = 3'b100;
            default: // ???
                controls = 3'b100;
        endcase

//    // Program Counter update for branch instructions
//    always @*
//        if(jump)
//            case(opcode[3])
//                1'b0: PC_select = 3'b100;            // jalr
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
//                default: PC_select = 3'b100;
//            endcase           
//        
//        // Compute memory offset for misaligned read/writes

endmodule