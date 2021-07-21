// Testbench for AND Gate
module and_gate_tb;

    // Declarations
    reg t_a, t_b;
    wire t_y;

    // Create instance of and_gate module
    and_gate test_and_gate(t_a, t_b, t_y);

    // Test
    initial
        begin
            $dumpfile("bin/and_gate_tb.vcd");
            $dumpvars(0, and_gate_tb);
            $monitor(t_a, t_b, t_y);

            // 0 AND 0
            t_a = 1'b0;
            t_b = 1'b0;

            // 0 AND 1
            #5 // 5 ns delay
            t_a = 1'b0;
            t_b = 1'b1;

            // 1 AND 0
            #5 // 5 ns delay
            t_a = 1'b1;
            t_b = 1'b0;

            // 1 AND 1
            #5 // 5 ns delay
            t_a = 1'b1;
            t_b = 1'b1;

            // 0 AND 0
            #5 // 5 ns delay
            t_a = 1'b0;
            t_b = 1'b0;
        end
endmodule