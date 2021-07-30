// Adder
module adder(a, b, sum);
    
    // Declarations
    input [31:0] a;
    input [31:0] b,
    output [31:0] sum;
    
    // Logic    
    assign sum = a + b;

endmodule

// Extend
module extend(instruction, imm_src, imm_ext);
    
    // Declarations
    input [31:7] instruction;
    input [1:0]  imm_src;
    output [31:0] imm_ext;

    // Logic    
    always_comb      
        case(immsrc)   
            // I−type          
            2'b00:immext = {{20{instr[31]}}, instr[31:20]};
            // S−type (stores)
            2'b01:immext = {{20{instr[31]}}, instr[31:25], instr[11:7]};
            // B−type (branches)
            2'b10:immext = {{20{instr[31]}}, instr[7], instr[30:25],  instr[11:8],  1’b0};
            // J−type (jal)
            2'b11:immext = {{12{instr[31]}}, instr[19:12], instr[20], instr[30:21], 1’b0};
            default: immext = 32'bx; // undefined
            
        endcase

endmodule

// Flip-Flop (w/reset)
module flopr #(parameter WIDTH = 8) (clock, reset, d, q);

    // Declarations
    input clock, 
    input reset,
    input [WIDTH−1:0] d; 
    output [WIDTH−1:0] q);   
    
    // Logic
    always_ff @(posedge clock, posedge reset)
        if (reset) q <= 0;
        else q <= d;

endmodule

// Flip-Flop (w/reset and enable)
module flopenr #(parameter WIDTH = 8) (clock, reset, enable, d, q);

    // Declarations
    input clock;
    input reset;
    input enable;
    input [WIDTH–1:0] d;
    output [WIDTH–1:0] q;
    
    // Logic
    always_ff @(posedge clk, posedge reset)
        if (reset)q <= 0;
        else if (en) q <= d;
    
endmodule

// Multiplexer (2:1)
module mux2 #(parameter WIDTH = 8) (d0, d1, s, y);
    
    // Declarations
    input [WIDTH−1:0] d0; 
    input [WIDTH−1:0] d1;
    input s;
    output [WIDTH−1:0] y;
    
    // Logic
    assign y = s ? d1 : d0;

endmodule

// Multiplexer (3:1)
module mux3 #(parameter WIDTH = 8) (d0, d1, d2, s, y);
    
    // Declarations
    input [WIDTH−1:0] d0;,
    input [WIDTH−1:0] d1;
    input [WIDTH−1:0] d2;
    input [1:0] s;
    output [WIDTH−1:0] y;
    
    // Logic
    assign y = s[1] ? d2 : (s[0] ? d1 : d0);

endmodule
