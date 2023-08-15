// Testbench for NBBPU
module nbbpu_tb;

    // Declarations
    reg t_clock;
    reg t_reset;
    wire [15:0] t_instruction;
    wire [15:0] t_data_in;
    wire t_data_write;
    wire [15:0] t_data_address;
    wire [15:0] t_data_out;
    wire [15:0] t_PC;

    // Debug    
    reg [7:0] instruction_counter;
    
    // Create instance of nbbpu module
    nbbpu test_nbbpu(t_clock, t_reset, t_instruction, t_data_in, t_data_write, t_data_address, t_data_out, t_PC);

    // Create instance of Instruction and Data Memory modules
    rom test_rom(t_PC, t_instruction);    
    //ram test_ram(clock, memory_control, data_adr, write_data, read_data);

    // Initialize
    initial
        begin
            $dumpfile("bin/nbbpu_tb.vcd");
            $dumpvars(0, nbbpu_tb);
            $monitor(t_clock, t_reset, t_instruction, t_data_in, t_data_write, t_data_address, t_data_out, t_PC);

            instruction_counter <= 0;
            t_reset <= 1; # 22; t_reset <= 0;
        end   

    // Generate clock
    always
        begin
            t_clock <= 1; # 5; t_clock <= 0; # 5;
        end
    
    // Test
    always @(negedge t_clock)
        begin
            instruction_counter <= instruction_counter + 1;
            if(instruction_counter >= 24)
                begin
                    $display("IC stopped");
                    $stop;
                end 
        end

endmodule