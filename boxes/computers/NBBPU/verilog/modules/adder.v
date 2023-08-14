// Adder
module adder(a, b, sum);
    
    // Declarations
    input [15:0] a;
    input [15:0] b;
    output [15:0] sum;
    
    // Logic    
    assign sum = a + b;

endmodule