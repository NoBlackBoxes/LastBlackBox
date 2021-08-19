// Testbench for Datapath
module datapath_tb;

    // Declarations
    reg t_clock;
    reg t_reset;
    reg [2:0] t_result_select;
    reg t_PC_select;
    reg t_ALU_select;
    reg t_reg_write;
    reg [2:0] t_ALU_control;
    reg [31:0] t_instruction;
    reg [31:0] t_read_data;
    wire t_zero;
    wire [31:0] t_PC;
    wire [31:0] t_ALU_result;
    wire [31:0] t_write_data;

    // Create instance of datapath module
    datapath test_datapath(t_clock, t_reset, t_result_select, t_PC_select, t_ALU_select, t_reg_write, t_ALU_control, t_instruction, t_read_data, t_zero, t_PC, t_ALU_result, t_write_data);

    // Test
    initial
        begin
            $dumpfile("bin/datapath_tb.vcd");
            $dumpvars(0, datapath_tb);
            $monitor(t_clock, t_reset, t_result_select, t_PC_select, t_ALU_select, t_reg_write, t_ALU_control, t_instruction, t_read_data, t_zero, t_PC, t_ALU_result, t_write_data);
            
            // Initialize
            t_clock <= 1'b0;
            t_reset <= 1'b0;
            #100; // 100 ns delay
        end

endmodule