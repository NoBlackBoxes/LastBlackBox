// Instruction memory module
module imem(a, rd);

    // Declarations
    input [31:0] a;
    output [31:0] rd;   

    // Intermediates
    logic [31:0] RAM[4096:0];

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
    output [31:0] rd;
    
    // Intermediates
    logic [31:0] RAM[4096:0];
    assign rd = RAM[a[31:2]]; // word aligned
    
    // Logic
    always @(posedge clock)
        if (we) RAM[a[31:2]] <= wd;

endmodule

// Testbench for CPU (RV32I)
module cpu_tb();

    // Intermediates
    reg clock;
    reg reset;
    wire [31:0] instruction;
    wire [31:0] read_data;
    wire mem_write;
    wire [31:0] PC;
    wire [31:0] data_adr;
    wire [31:0] write_data;
    
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
            $dumpvars(0, cpu_tb);
            $monitor(clock, reset, instruction, read_data, mem_write, PC, data_adr, write_data);

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
            if(mem_write) 
                begin
                    if(data_adr === 32'hFFFFFFF0 & write_data === 1) 
                        begin
                            $display("Verification succeeded");
                            $stop;
                        end 
                    else if (data_adr === 32'hFFFFFFF0)
                        begin
                            $display("Verification failed");
                            $stop;
                        end
                end   
        end

endmodule