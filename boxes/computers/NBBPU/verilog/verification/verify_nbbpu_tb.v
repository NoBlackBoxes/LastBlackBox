// Testbench for verifying NBBPU
module verify_nbbpu_tb();

    // Declarations
    reg t_clock;
    reg t_reset;
    wire t_select;
    wire [15:0] t_instruction;
    wire [15:0] t_read_data;
    wire t_instruction_enable;
    wire t_read_enable;
    wire t_write_enable;
    wire [15:0] t_address;
    wire [15:0] t_write_data;
    wire [15:0] t_PC;
    wire t_debug;

    // Assignments
    assign t_select = 1'b1;

    // Debug    
    reg [20:0] instruction_counter;

    // Create instance of nbbpu module
    nbbpu test_nbbpu(
                        t_clock, 
                        t_reset, 
                        t_instruction, 
                        t_read_data, 
                        t_instruction_enable, 
                        t_read_enable, 
                        t_write_enable, 
                        t_address, 
                        t_write_data, 
                        t_PC, 
                        t_debug
                    );
    
    // Create instance of Instruction and Data Memory modules
    rom test_rom(t_clock, t_select, t_instruction_enable, t_PC, t_instruction);
    ram test_ram(t_clock, t_select, t_read_enable, t_write_enable, t_address, t_write_data, t_read_data);

    // Initialize
    initial
        begin
            $dumpfile("bin/verify_nbbpu_tb.vcd");
            $dumpvars(0, verify_nbbpu_tb);
            
            instruction_counter <= 0;
            t_reset <= 0; # 22; t_reset <= 1; # 20; t_reset <= 0;
        end   
    
    // Generate clock
    always
        begin
            t_clock <= 1; # 5; t_clock <= 0; # 5;
        end   
    
    // check results
    always @(negedge t_clock)
        begin
            instruction_counter <= instruction_counter + 1;
            if(instruction_counter >= 32'h00000FFF)
                begin
                    $display("IC stopped");
                    $stop;
                end 
            if(t_write_enable) 
                begin
                    if(t_address === 16'hFFF0 & t_write_data === 1) 
                        begin
                            $write("%c[1;32m",27);
                            $display(" - Verification succeeded");
                            $write("%c[0m",27);
                            $finish;
                        end 
                    else if (t_address === 16'hFFF0)
                        begin
                            $write("%c[1;31m",27);
                            $display(" ! Verification failed");
                            $write("%c[0m",27);
                            $stop;
                        end
                end   
        end

endmodule