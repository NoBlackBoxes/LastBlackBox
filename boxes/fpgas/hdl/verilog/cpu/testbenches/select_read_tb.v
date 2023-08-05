// Testbench for Select Read
module select_read_tb;

    // Declarations
    reg [2:0] t_funct3;
    reg [31:0] t_read_data_in;
    wire [31:0] t_read_data_out;

    // Create instance of generate immediate module
    select_read test_select_read(t_funct3, t_read_data_in, t_read_data_out);

    // Test
    initial
        begin
            $dumpfile("bin/select_read_tb.vcd");
            $dumpvars(0, select_read_tb);
            $monitor(t_funct3, t_read_data_in, t_read_data_out);

            // lb
            t_funct3 <= 3'b000;
            t_read_data_in <= 32'hF2F4F6F8;

            // lh
            #100 // 100 ns delay
            t_funct3 <= 3'b001;
            
            // lw
            #100 // 100 ns delay
            t_funct3 <= 3'b010;

            // lbu
            #100 // 100 ns delay
            t_funct3 <= 3'b100;

            // lhu
            #100 // 100 ns delay
            t_funct3 <= 3'b101;

            // Wait
            #100 // 100 ns delay
            ;
        end

endmodule