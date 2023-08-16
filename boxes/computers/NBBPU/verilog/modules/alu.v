// ALU
module alu(X, Y, opcode, read_data, Z);
    
    // Declarations
    input [15:0] X;
    input [15:0] Y;
    input [3:0] opcode;
    input [15:0] read_data;
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
            4'b1100: Z <= read_data;        // (memory operation) load data
            4'b1101: Z <= Y;                // (memory operation) store data
            default: Z <= 15'd0;            // Output Zero

        endcase

endmodule