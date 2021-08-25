// ALU
module alu(src_A, src_B, ALU_control, ALU_result, zero);
    
    // Declarations
    input [31:0] src_A;
    input [31:0] src_B;
    input [3:0] ALU_control;
    output reg [31:0] ALU_result;
    output zero;

    // Logic
    always @*
        case(ALU_control)
            4'b0000: ALU_result <= src_A + src_B; // addition   
            4'b0001: ALU_result <= src_A - src_B; // subtraction   
            4'b0010: ALU_result <= src_A & src_B; // and
            4'b0011: ALU_result <= src_A | src_B; // or
            4'b0100: ALU_result <= src_A ^ src_B; // xor
            4'b0101: ALU_result <= ($signed(src_A) < $signed(src_B)) ? 1 : 0; // set less than (signed)
            4'b0110: ALU_result <= ($unsigned(src_A) < $unsigned(src_B)) ? 1 : 0; // set less than (unsigned)
            4'b0111: ALU_result <= $signed(src_A) >>> src_B[4:0]; // shift right (arithmetic)

            // Need to distinguish shifts (right/left and artihmetic/logical)

        endcase
    assign zero = (ALU_result == 0) ? 1 : 0; // Set zero flag

endmodule