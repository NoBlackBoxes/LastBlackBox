// Adder
module adder(a, b, sum);
    
    // Declarations
    input [31:0] a;
    input [31:0] b;
    output [31:0] sum;
    
    // Logic    
    assign sum = a + b;

endmodule

// Extend
module extend(instruction, imm_src, imm_ext);
    
    // Declarations
    input [31:7] instruction;
    input [1:0] imm_src;
    output reg [31:0] imm_ext;

    // Logic
    always @*
        begin
            case(imm_src)
                2'b00: imm_ext = {{20{instruction[31]}}, instruction[31:20]}; // I−type
                2'b01: imm_ext = {{20{instruction[31]}}, instruction[31:25], instruction[11:7]}; // S−type (stores)
                2'b10: imm_ext = {{20{instruction[31]}}, instruction[7], instruction[30:25],  instruction[11:8], 1'b0}; // B−type (branches)
                2'b11: imm_ext = {{12{instruction[31]}}, instruction[19:12], instruction[20], instruction[30:21], 1'b0}; // J−type (jal)
                default: imm_ext = 32'bx; // Undefined
            endcase
        end

endmodule

// Flip-Flop (w/reset)
module flopr(clock, reset, d, q);

    // Parameters
    parameter WIDTH = 8;
    
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
