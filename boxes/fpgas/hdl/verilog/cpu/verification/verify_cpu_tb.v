// Instruction memory module
module imem(a, rd);

    // Declarations
    input [31:0] a;
    output [31:0] rd;   

    // Intermediates
    reg [31:0] RAM[0:4095];

    // Logic
    initial
        $readmemh("bin/imem.txt", RAM);
    
    assign rd = RAM[a[31:2]]; // word aligned

endmodule

// Data memory module
module dmem(clock, we, a, wd, rd);

    // Declarations
    input clock;
    input we;
    input [31:0] a;
    input [31:0] wd;
    output reg [31:0] rd;
    
    // Intermediates
    wire misaligned;
    reg [7:0] RAM[0:4095];
    
    // Initialize
    initial
        $readmemh("bin/dmem.txt", RAM);

    // Logic (control)
    assign misaligned = a[0] | a[1];
    assign sub_byte = a[1:0];

    // Logic (read)
    always @*
        begin
            if (misaligned)
                rd = {24'h000000, RAM[a]};
            else
                rd = {RAM[a+3], RAM[a+2], RAM[a+1], RAM[a+0]};
        end

    // Logic (write)
    always @(posedge clock)
        if (we) RAM[a[31:2]] <= wd;

endmodule

// Testbench for verifying CPU (RV32I)
module verify_cpu_tb();

    // Intermediates
    reg clock;
    reg reset;
    wire [31:0] instruction;
    wire [31:0] read_data;
    wire mem_write;
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
        mem_write, 
        PC, 
        data_adr, 
        write_data);
    
    // Create instance of Instruction and Data Memory modules
    imem imem(PC, instruction);    
    dmem dmem(clock, mem_write, data_adr, write_data, read_data);

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
            if(mem_write) 
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