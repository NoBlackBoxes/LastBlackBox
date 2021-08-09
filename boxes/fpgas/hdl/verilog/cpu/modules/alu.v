// ALU
module alu(src_A, src_B, ALU_control, ALU_result, zero);
    
    // Declarations
    input [31:0] src_A;
    input [31:0] src_B;
    input [2:0] ALU_control;
    output reg [31:0] ALU_result;
    output zero;

    // Logic
    always @*
        case(ALU_control)
            3'b000: ALU_result <= src_A + src_B; // addition   
            3'b001: ALU_result <= src_A - src_B; // subtraction   
            3'b010: ALU_result <= src_A & src_B; // and
            3'b011: ALU_result <= src_A | src_B; // or
            3'b101: ALU_result <= ($signed(src_A) < $signed(src_B)) ? 1 : 0; // set less than   
        endcase

endmodule