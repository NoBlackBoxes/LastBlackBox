// Multiplexer (3:1)
module mux3(d0, d1, d2, s, y);
    
    // Parameters
    parameter WIDTH = 8;

    // Declarations
    input [WIDTH-1:0] d0; 
    input [WIDTH-1:0] d1;
    input [WIDTH-1:0] d2;
    input [1:0] s;
    output [WIDTH-1:0] y;
    
    // Logic
    assign y = s[1] ? d2 : (s[0] ? d1 : d0);

endmodule