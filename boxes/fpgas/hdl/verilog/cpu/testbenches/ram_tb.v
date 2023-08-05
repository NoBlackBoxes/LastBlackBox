// Testbench for RAM
module ram_tb;

    // Declarations
    reg t_clock;
    reg [3:0] t_control;
    reg [31:0] t_address;
    reg [31:0] t_write_data;
    wire [31:0] t_read_data;

    // Create instance of ram module
    ram test_ram(t_clock, t_control, t_address, t_write_data, t_read_data);

    // Test
    initial
        begin
            $dumpfile("bin/ram_tb.vcd");
            $dumpvars(0, ram_tb);
            $monitor(t_clock, t_control, t_address, t_write_data, t_read_data);

            // Read Address 0
            #100 // 100 ns delay
            t_clock <= 1'b0;
            t_control <= 3'b000;
            t_address <= 32'h00000000;
            t_write_data <= 32'h00000000;
            #100 // 100 ns delay
            t_clock <= 1'b1;
            
            // Write Address 1
            #100 // 100 ns delay
            t_clock <= 1'b0;
            t_control <= 3'b001;
            t_address <= 32'h00000001;
            t_write_data <= 32'h0000000F;
            #100 // 100 ns delay
            t_clock <= 1'b1;

            // Read Address 1
            #100 // 100 ns delay
            t_clock <= 1'b0;
            t_control <= 3'b000;
            t_address <= 32'h00000001;
            t_write_data <= 32'h00000000;
            #100 // 100 ns delay
            t_clock <= 1'b1;

            // Wait
            #100 // 100 ns delay
            t_control <= 3'b000;
            t_clock <= 1'b0;
            #100 // 100 ns delay
            t_clock <= 1'b1;
            #100 // 100 ns delay
            t_clock <= 1'b0;
            ;
        end

endmodule