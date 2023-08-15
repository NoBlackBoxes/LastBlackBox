// ALU
module alu(X, Y, opcode, Z);
    
    // Declarations
    input [15:0] X;
    input [15:0] Y;
    input [3:0] opcode;
    output reg [15:0] Z;

    // Logic
    always @*
        case(opcode)
            4'b0000: Z <= X + Y;            // addition   
            4'b0001: Z <= X - Y;            // subtraction   
            4'b0010: Z <= X & Y;            // and
            4'b0011: Z <= X | Y;            // or
            4'b0100: Z <= X ^ Y;            // xor
            4'b0101: Z <= X >> Y[3:0];      // shift right (logical)
            4'b0110: Z <= X << Y[3:0];      // shift left (logical)
            4'b0101: Z <= X >= Y ? 1 : 0;   // compare (greater or equal)

        endcase

endmodule