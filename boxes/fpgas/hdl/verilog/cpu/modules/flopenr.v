// Flip-Flop (w/reset and enable)
module flopenr(clock, reset, enable, d, q);

    // Parameters
    parameter WIDTH = 8;

    // Declarations
    input clock;
    input reset;
    input enable;
    input [WIDTH-1:0] d;
    output reg [WIDTH-1:0] q;
    
    // Logic
    always @(posedge clock, posedge reset)
        if (reset)
            q <= 0;
        else if (enable)
            q <= d;
    
endmodule