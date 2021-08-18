// Main Decoder
module main_decoder(opcode, reg_write, ALU_select, mem_write, result_select, branch, ALU_op, jump);

    // Declarations
    input [6:0] opcode;
    output reg_write;
    output ALU_select;
    output mem_write;
    output [1:0] result_select;
    output branch;
    output [1:0] ALU_op;
    output jump;
    
    // Intermediates
    reg [8:0] controls;
    assign {reg_write, ALU_select, mem_write, result_select, branch, ALU_op, jump} = controls;   
    
    // Logic
    always @*
        case(opcode)      
            // reg_write - ALU_select - mem_write - result_select - branch - ALU_op - jump
            7'b0000011: controls = 9'b1_1_0_01_0_00_0; // lw
            7'b0100011: controls = 9'b0_1_1_00_0_00_0; // sw
            7'b0110011: controls = 9'b1_0_0_00_0_10_0; // R–type
            7'b0110111: controls = 9'b1_0_0_11_0_00_0; // lui
            7'b1100011: controls = 9'b0_0_0_00_1_01_0; // beq
            7'b1100011: controls = 9'b0_0_0_00_1_01_0; // bne
            7'b0010011: controls = 9'b1_1_0_00_0_10_0; // I–type ALU
            7'b1101111: controls = 9'b1_0_0_10_0_00_1; // jal
            default:    controls = 9'bx_x_x_xx_x_xx_x; // ???   
        endcase

endmodule