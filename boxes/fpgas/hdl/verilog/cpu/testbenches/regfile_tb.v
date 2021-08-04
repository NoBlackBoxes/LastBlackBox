// Testbench for Adder
module regfile_tb;

    // Declarations
    reg t_clock;
    reg t_write_enable;
    reg [5:0] t_address_read_1;
    reg [5:0] t_address_read_2;
    reg [5:0] t_address_write;
    reg [31:0] t_write_data;
    wire [31:0] t_read_data_1;
    wire [31:0] t_read_data_2;

    // Create instance of adder module
    regfile test_regfile(t_clock, t_write_enable, t_address_read_1, t_address_read_2, t_address_write, t_write_data, t_read_data_1, t_read_data_2);

    // Intermediates
    integer  i;

    // Test
    initial
        begin
            $dumpfile("bin/regfile_tb.vcd");
            $dumpvars(0, regfile_tb);
            $monitor(t_clock, t_write_enable, t_address_read_1, t_address_read_2, t_address_write, t_write_data, t_read_data_1, t_read_data_2);
            
            // Initialize registers
            #100; // 100 ns delay
            for (i = 0; i < 32; i = i + 1)
                begin
                    t_clock <= 1'b0;
                    t_write_enable <= 1'b1;
                    t_address_write <= i;
                    t_write_data <= i + 100;
                    #10; // 10 ns delay
                    t_clock = 1'b1;
                end

            // Read registers 1 and 2
            t_clock <= 1'b0;
            t_write_enable <= 1'b0;
            t_address_read_1 <= 5'b00001;
            t_address_read_2 <= 5'b00010;
            #100; // 100 ns delay

            // Read registers 4 and 5
            t_clock <= 1'b0;
            t_write_enable <= 1'b0;
            t_address_read_1 <= 5'b00100;
            t_address_read_2 <= 5'b00101;
            #100; // 100 ns delay

            // Read registers 0 and 31
            t_clock <= 1'b0;
            t_write_enable <= 1'b0;
            t_address_read_1 <= 5'b00000;
            t_address_read_2 <= 5'b11111;
            #100; // 100 ns delay

        end

endmodule