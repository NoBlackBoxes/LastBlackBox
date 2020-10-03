// Testbench for AND Gate
module and_gate_tb;

wire t_Y;
reg t_A, t_B;

// Create instance of and_gate module
and_gate test_gate(t_Y, t_A, t_B);

initial
    begin
        $dumpfile("bin/and_gate_tb.vcd");
        $dumpvars(0,and_gate_tb);
        $monitor(t_A, t_B, t_Y);

        t_A = 1'b0;
        t_B = 1'b0;

        #5 // 5 ns delay
        t_A = 1'b0;
        t_B = 1'b1;

        #5 // 5 ns delay
        t_A = 1'b1;
        t_B = 1'b0;

        #5 // 5 ns delay
        t_A = 1'b1;
        t_B = 1'b1;

        #5 // 5 ns delay
        t_A = 1'b0;
        t_B = 1'b0;

    end
endmodule