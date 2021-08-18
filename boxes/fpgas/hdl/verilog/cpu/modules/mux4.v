// Multiplexer (4:1)
module mux4(d0, d1, d2, d3, s, y);
    
    // Parameters
    parameter WIDTH = 8;

    // Declarations
    input [WIDTH-1:0] d0;
    input [WIDTH-1:0] d1;
    input [WIDTH-1:0] d2;
    input [WIDTH-1:0] d3;
    input [1:0] s;
    output [WIDTH-1:0] y;
    
    // Logic
    assign y = s[1] ? (s[0] ? d3 : d2) : (s[0] ? d1 : d0);

endmodule