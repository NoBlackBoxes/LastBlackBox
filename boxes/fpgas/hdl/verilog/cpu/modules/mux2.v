// Multiplexer (2:1)
module mux2(d0, d1, s, y);
    
    // Parameters
    parameter WIDTH = 8;

    // Declarations
    input [WIDTH-1:0] d0; 
    input [WIDTH-1:0] d1;
    input s;
    output [WIDTH-1:0] y;
    
    // Logic
    assign y = s ? d1 : d0;

endmodule