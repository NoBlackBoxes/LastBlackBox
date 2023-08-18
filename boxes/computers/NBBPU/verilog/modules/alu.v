// ALU
module alu(X, Y, instruction, read_data, PC_plus1, Z);
    
    // Declarations
    input [15:0] X;
    input [15:0] Y;
    input [15:0] instruction;
    input [15:0] read_data;
    input [15:0] PC_plus1;
    output reg [15:0] Z;

    // Intermediates
    wire [3:0] opcode;
    wire [15:0] lower_byte, upper_byte;
    assign opcode = instruction[15:12];
    assign lower_byte = {X[15:8], instruction[11:4]};
    assign upper_byte = {instruction[11:4], X[7:0]};

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
            4'b0111: Z <= X >= Y ? 1 : 0;   // compare (greater or equal)
            4'b1000: Z <= PC_plus1;         // (control operation) jump
            4'b1001: Z <= Y == 0 ? 1 : 0;   // (control operation) branch if equal
            4'b1010: Z <= Y != 0 ? 1 : 0;   // (control operation) branch if not equal
            4'b1011: Z <= 15'd0;            // (reserved operation)
            4'b1100: Z <= read_data;        // (memory operation) load data
            4'b1101: Z <= Y;                // (memory operation) store data
            4'b1110: Z <= lower_byte;       // (memory operation) set register lower byte
            4'b1111: Z <= upper_byte;       // (memory operation) set register upper byte
            default: Z <= 15'd0;            // Output Zero

        endcase

endmodule