// Full Adder
module full_adder (a, b, in_carry, sum, out_carry);

    // Declarations
    input  a;
    input  b;
    input  in_carry;
    output sum;
    output out_carry;
    
    // Intermediates
    wire   x;
    wire   y;
    wire   z;

    // Logic
    assign x = a ^ b;               // a XOR b
    assign y = x & in_carry;        // x AND input carry
    assign z = a & b;               // a AND b
    assign sum   = x ^ in_carry;    // x XOR input carry
    assign out_carry = y | z;       // y OR z
   
endmodule