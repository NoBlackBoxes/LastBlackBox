// Adder
module adder(a, b, sum);
    
    // Declarations
    input [31:0] a;
    input [31:0] b;
    output [31:0] sum;
    
    // Logic    
    assign sum = a + b;

endmodule