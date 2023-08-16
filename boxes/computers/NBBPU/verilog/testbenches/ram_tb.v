// Testbench for RAM
module ram_tb;

    // Declarations
    reg t_clock;
    reg t_write_enable;
    reg [15:0] t_address;
    reg [15:0] t_write_data;
    wire [15:0] t_read_data;

    // Create instance of ram module
    ram test_ram(t_clock, t_write_enable, t_address, t_write_data, t_read_data);

    // Test
    initial
        begin
            $dumpfile("bin/ram_tb.vcd");
            $dumpvars(0, ram_tb);
            $monitor(t_clock, t_write_enable, t_address, t_write_data, t_read_data);

            // Read Address 0
            #100 // 100 ns delay
            t_clock <= 1'b0;
            t_write_enable <= 1'b0;
            t_address <= 16'h00000000;
            t_write_data <= 16'h00000000;
            #100 // 100 ns delay
            t_clock <= 1'b1;
            
            // Write Address 1
            #100 // 100 ns delay
            t_clock <= 1'b0;
            t_write_enable <= 1'b1;
            t_address <= 16'h00000001;
            t_write_data <= 16'h0000000F;
            #100 // 100 ns delay
            t_clock <= 1'b1;

            // Read Address 1
            #100 // 100 ns delay
            t_clock <= 1'b0;
            t_write_enable <= 1'b0;
            t_address <= 16'h00000001;
            t_write_data <= 16'h00000000;
            #100 // 100 ns delay
            t_clock <= 1'b1;

            // Wait
            #100 // 100 ns delay
            t_write_enable <= 1'b0;
            t_clock <= 1'b0;
            #100 // 100 ns delay
            t_clock <= 1'b1;
            #100 // 100 ns delay
            t_clock <= 1'b0;
            ;
        end

endmodule