// Testbench for SR Latch
module sr_latch_tb;

wire t_Q, t_Qn;
reg t_S, t_R;

// Create instance of sr_latch module
sr_latch test_gate(t_Q, t_Qn, t_S, t_R);

initial
    begin
        $dumpfile("bin/sr_latch_tb.vcd");
        $dumpvars(0,sr_latch_tb);
        $monitor(t_Q, t_Qn, t_S, t_R);

        t_S = 1'b0;
        t_R = 1'b0;

        #5 // 5 ns delay
        t_S = 1'b0;
        t_R = 1'b1;

        #5 // 5 ns delay
        t_S = 1'b1;
        t_R = 1'b0;

        #5 // 5 ns delay
        t_S = 1'b1;
        t_R = 1'b1;

        #5 // 5 ns delay
        t_S = 1'b0;
        t_R = 1'b0;

    end
endmodule