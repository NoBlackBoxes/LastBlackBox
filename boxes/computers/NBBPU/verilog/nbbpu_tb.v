// Testbench for NBBPU
module nbbpu_tb;

    // Declarations
    reg t_clock;
    reg t_reset;
    reg [15:0] t_instruction;
    reg [15:0] t_data_in;
    wire [15:0] t_PC;
    wire [3:0] t_memory_control;
    wire [15:0] t_data_out;

    // Create instance of nbbpu module
    nbbpu test_nbbpu(t_clock, t_reset, t_instruction, t_data_in, t_PC, t_memory_control, t_data_out);

    // Create clock
    always #5 t_clock = ~t_clock;

    // Test
    initial
        begin
            $dumpfile("bin/nbbpu_tb.vcd");
            $dumpvars(0, nbbpu_tb);
            $monitor(t_clock, t_reset, t_PC);

            // Initial
            t_clock <= 1'b0;
            t_reset <= 1'b0;

            // Initial: Reset
            t_reset <= 1'b1;
            #10 // 100 ns delay
            t_reset <= 1'b0;

            // Wait
            #200 // 100 ns delay
            ;

            // Final: Reset
            t_reset = 1'b1;
            #10 // 100 ns delay
            t_reset = 1'b0;

            // Finish
            #100 $finish; // 100 ns delay    
        end

endmodule