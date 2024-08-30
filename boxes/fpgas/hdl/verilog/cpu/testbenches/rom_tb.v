// Testbench for ROM
module rom_tb;

    // Declarations
    reg [31:0] t_address;
    wire [31:0] t_read_data;

    // Create instance of rom module
    rom test_rom(t_address, t_read_data);

    // Test
    initial
        begin
            $dumpfile("bin/rom_tb.vcd");
            $dumpvars(0, rom_tb);
            $monitor(t_address, t_read_data);

            // Address 0
            #100 // 100 ns delay
            t_address <= 32'h00000000;
            
            // Address 4 (2nd value)
            #100 // 100 ns delay
            t_address <= 32'h00000004;

            // Address 8 (3rd value)
            #100 // 100 ns delay
            t_address <= 32'h00000008;

            // Address 256 (64th value)
            #100 // 100 ns delay
            t_address <= 32'h00000100;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule