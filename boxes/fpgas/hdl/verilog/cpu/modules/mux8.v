// Multiplexer (8:1)
module mux8(d0, d1, d2, d3, d4, d5, d6, d7, s, y);
    
    // Parameters
    parameter WIDTH = 8;

    // Declarations
    input [WIDTH-1:0] d0;
    input [WIDTH-1:0] d1;
    input [WIDTH-1:0] d2;
    input [WIDTH-1:0] d3;
    input [WIDTH-1:0] d4;
    input [WIDTH-1:0] d5;
    input [WIDTH-1:0] d6;
    input [WIDTH-1:0] d7;
    input [2:0] s;
    output reg [WIDTH-1:0] y;
    
    // Logic
    always @*
        begin
            case(s)
                3'b000: y = d0;
                3'b001: y = d1;
                3'b010: y = d2;
                3'b011: y = d3;
                3'b100: y = d4;
                3'b101: y = d5;
                3'b110: y = d6;
                3'b111: y = d7;
                default: y = d0;
            endcase
        end

endmodule