// Main Decoder
module main_decoder(opcode, reg_write, immediate_select, ALU_select, mem_write, result_select, branch, ALU_op, jump);

    // Declarations
    input [6:0] opcode;
    output reg_write;
    output [1:0] immediate_select;
    output ALU_select;
    output mem_write;
    output [1:0] result_select;
    output branch;
    output [1:0] ALU_op;
    output jump;
    
    // Intermediates
    reg [10:0] controls;
    assign {reg_write, immediate_select, ALU_select, mem_write, result_select, branch, ALU_op, jump} = controls;   
    
    // Logic
    always @*
        case(opcode)      
            // reg_write - immediate_select - ALU_select - mem_write - result_select - branch - ALU_op - jump
            7'b0000011: controls = 11'b1_00_1_0_01_0_00_0; // lw
            7'b0100011: controls = 11'b0_01_1_1_00_0_00_0; // sw
            7'b0110011: controls = 11'b1_xx_0_0_00_0_10_0; // R–type
            7'b1100011: controls = 11'b0_10_0_0_00_1_01_0; // beq
            7'b0010011: controls = 11'b1_00_1_0_00_0_10_0; // I–type ALU
            7'b1101111: controls = 11'b1_11_0_0_10_0_00_1; // jal
            default:    controls = 11'bx_xx_x_x_xx_x_xx_x; // ???   
        endcase

endmodule