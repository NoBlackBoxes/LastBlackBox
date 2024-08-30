// Testbench for Generate Immediate
module generate_immediate_tb;

    // Declarations
    reg [31:0] t_instruction;
    wire [31:0] t_immediate;

    // Create instance of generate immediate module
    generate_immediate test_generate_immediate(t_instruction, t_immediate);

    // Test
    initial
        begin
            $dumpfile("bin/generate_immediate_tb.vcd");
            $dumpvars(0, generate_immediate_tb);
            $monitor(t_instruction, t_immediate);

            // Initial
            t_instruction <= 32'h00000000;

            // Generate lw
            #100 // 100 ns delay
            t_instruction <= 32'h06002103;
            
            // Generate lui
            #100 // 100 ns delay
            t_instruction <= 32'hFFFFF037;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule