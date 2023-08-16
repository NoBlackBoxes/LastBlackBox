// Testbench for NBBPU
module nbbpu_tb;

    // Declarations
    reg t_clock;
    reg t_reset;
    wire [15:0] t_instruction;
    wire [15:0] t_read_data;
    wire t_write_enable;
    wire [15:0] t_address;
    wire [15:0] t_write_data;
    wire [15:0] t_PC;

    // Debug    
    reg [7:0] instruction_counter;
    
    // Create instance of nbbpu module
    nbbpu test_nbbpu(t_clock, t_reset, t_instruction, t_read_data, t_write_enable, t_address, t_write_data, t_PC);

    // Create instance of Instruction and Data Memory modules
    rom test_rom(t_PC, t_instruction);
    ram test_ram(t_clock, t_write_enable, t_address, t_write_data, t_read_data);

    // Initialize
    initial
        begin
            $dumpfile("bin/nbbpu_tb.vcd");
            $dumpvars(0, nbbpu_tb);
            $monitor(t_clock, t_reset, t_instruction, t_read_data, t_write_enable, t_address, t_write_data, t_PC);

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