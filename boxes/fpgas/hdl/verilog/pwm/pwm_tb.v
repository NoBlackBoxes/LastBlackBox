// Testbench for PWM
module pwm_tb;

    // Declarations
    reg t_clock;
    reg [7:0] t_duty_cycle;
    wire t_pulse;

    // Create instance of pwm module
    pwm test_pwm(t_clock, t_duty_cycle, t_pulse);

    // Create clock
    always #1 t_clock = ~t_clock;

    // Test
    initial
        begin
            $dumpfile("bin/pwm_tb.vcd");
            $dumpvars(0, pwm_tb);
            $monitor(t_clock, t_duty_cycle, t_pulse);

            // Initial
            t_clock <= 1'b0;
            t_duty_cycle <= 8'hFF;
            #100; // 100 ns delay

            // 16/256
            t_duty_cycle <= 8'h0F;
            #1000; // 100 ns delay

            // 8/256
            t_duty_cycle <= 8'h08;
            #1000; // 100 ns delay

            // 2/256
            t_duty_cycle <= 8'h02;
            #1000; // 100 ns delay

            // Finish
            #100 $finish; // 100 ns delay    
        end

endmodule