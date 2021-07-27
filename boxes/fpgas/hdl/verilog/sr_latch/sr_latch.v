// SR Latch
`timescale 1ns / 1ps
module sr_latch(s, r, q, qn);
 
    // Declarations
    input s;
    input r;
    output q;
    output qn;
    
    // Intermediates
    wire q_internal;
    wire qn_internal;
 
    // Logic
    assign #1 q_internal = ~(s & qn_internal);
    assign #1 qn_internal = ~(r & q_internal);
    assign q = q_internal;
    assign qn = qn_internal;
 
endmodule