// SR Latch
`timescale 1ns / 1ps
module sr_latch(Q, Qn, S, R);
 
    output Q, Qn;
    input S, R;
    
    wire Q_internal, Qn_internal;
 
    assign #1 Q_internal = ~(S & Qn_internal);
    assign #1 Qn_internal = ~(R & Q_internal);
    assign Q = Q_internal;
    assign Qn = Qn_internal;
 
endmodule