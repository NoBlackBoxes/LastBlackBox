// Counter (4-bit)
`timescale 1ns / 1ps
module counter(clock, reset, count);
 
    // Declarations
    input clock;
    input reset;
    output reg[3:0] count;
    
    // Intermediates
    wire q_internal;
    wire qn_internal;
 
    // Logic
    always @(posedge clock)
        begin
            if(!reset)
                count <= 0;
            else
                count <= count + 1;
        end 
endmodule