// Testbench for verifying CPU (RV32I)
module verify_cpu_tb();

    // Intermediates
    reg clock;
    reg reset;
    wire [31:0] instruction;
    wire [31:0] read_data;
    wire [3:0] memory_control;
    wire [31:0] PC;
    wire [31:0] data_adr;
    wire [31:0] write_data;

    // Debug    
    reg [9:0] instruction_counter;

    // Create instance of CPU module
    cpu test_cpu(
        clock, 
        reset, 
        instruction, 
        read_data, 
        memory_control, 
        PC, 
        data_adr, 
        write_data);
    
    // Create instance of Instruction and Data Memory modules
    rom rom(PC, instruction);    
    ram ram(clock, memory_control, data_adr, write_data, read_data);

    // initialize test
    initial
        begin
            $dumpfile("bin/verify_cpu_tb.vcd");
            $dumpvars(0, verify_cpu_tb);
            
            instruction_counter <= 0;
            reset <= 1; # 22; reset <= 0;
        end   
    
    // generate clock to sequence tests
    always
        begin
            clock <= 1; # 5; clock <= 0; # 5;
        end   
    
    // check results
    always @(negedge clock)
        begin
            instruction_counter <= instruction_counter + 1;
            if(instruction_counter >= 1023)
                begin
                    $display("IC stopped");
                    $stop;
                end 
            if(memory_control[0]) 
                begin
                    if(data_adr === 32'hFFFFFFF0 & write_data === 1) 
                        begin
                            $display(" - Verification succeeded");
                            $finish;
                        end 
                    else if (data_adr === 32'hFFFFFFF0)
                        begin
                            $display(" ! Verification failed");
                            $stop;
                        end
                end   
        end

endmodule