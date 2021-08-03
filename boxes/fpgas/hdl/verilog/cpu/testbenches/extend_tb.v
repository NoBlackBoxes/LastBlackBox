// Testbench for Extend
module extend_tb;

    // Declarations
    reg [31:0] t_instruction;
    reg [1:0] t_immediate_select;
    wire [31:0] t_immediate_extended;

    // Create instance of extend module
    extend test_extend(t_instruction, t_immediate_select, t_immediate_extended);

    // Test
    initial
        begin
            $dumpfile("bin/extend_tb.vcd");
            $dumpvars(0, extend_tb);
            $monitor(t_instruction, t_immediate_select, t_immediate_extended);

            // Initial
            t_instruction <= 32'h00000000;
            t_immediate_select <= 2'b00;

            // Extend lw
            #100 // 100 ns delay
            t_instruction <= 32'h06002103;
            t_immediate_select <= 2'b00;
            
            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule