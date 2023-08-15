// Flip-Flop (w/reset)
module flopr(clock, reset, d, q);

    // Parameters
    parameter WIDTH = 16;
    
    // Declarations
    input clock; 
    input reset;
    input [WIDTH-1:0] d; 
    output reg [WIDTH-1:0] q;   
    
    // Logic
    always @(posedge clock, posedge reset)
        if (reset)
            q <= 0;
        else 
            q <= d;

endmodule