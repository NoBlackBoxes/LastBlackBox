// Testbench for Full Adder
module full_adder_tb;

    // Declarations
    reg  t_a;
    reg  t_b;
    reg  t_in_carry;
    wire t_sum;
    wire t_out_carry;

    // Create instance of full_adder module
    full_adder test_full_adder(t_a, t_b, t_in_carry, t_sum, t_out_carry);

    // Test
    initial
        begin
            $dumpfile("bin/full_adder_tb.vcd");
            $dumpvars(0, full_adder_tb);
            $monitor(t_a, t_b, t_in_carry, t_sum, t_out_carry);

            // 0 + 0
            t_a = 1'b0;
            t_b = 1'b0;
            t_in_carry = 1'b0;

            // 0 + 1
            #5 // 5 ns delay
            t_a = 1'b0;
            t_b = 1'b1;
            t_in_carry = 1'b0;

            // 1 + 1
            #5 // 5 ns delay
            t_a = 1'b1;
            t_b = 1'b1;
            t_in_carry = 1'b0;

            // 0 + 0
            #5 // 5 ns delay
            t_a = 1'b0;
            t_b = 1'b0;
            t_in_carry = 1'b0;
        end

endmodule