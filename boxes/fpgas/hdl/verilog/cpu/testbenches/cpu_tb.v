// Instruction memory module
module imem();

    // Declarations
    input [31:0] a;
    output [31:0] rd;   

    // Intermediates
    logic [31:0] RAM[63:0];

    // Logic
    initial
        $readmemh("cpu_test.txt", RAM);
    
    assign rd = RAM[a[31:2]]; // word aligned

endmodule

// Data memory module
module dmem();

    // Declarations
    input clock;
    input we;
    input [31:0] a; 
    input [31:0] wd;
    output [31:0] rd;
    
    // Intermediates
    logic [31:0] RAM[63:0];
    assign rd = RAM[a[31:2]]; // word aligned
    
    // Logic
    always_ff @(posedge clk)
        if (we) RAM[a[31:2]] <= wd;

endmodule

// Testbench for CPU (RV32I)
module testbench();

    // Intermediates
    wire clock;
    wire reset;
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
            reset <= 1; # 22; reset <= 0;
        end   
    
    // generate clock to sequence tests
    always
        begin
            clock <= 1; # 5; clk <= 0; # 5;
        end   
    
    // check results
    always @(negedge clk)
        begin
            if(MemWrite) 
                begin
                    if(DataAdr === 100 & WriteData === 25) 
                        begin
                            $display("Simulation succeeded");
                            $stop;
                        end 
                    else if (DataAdr !== 96)
                        begin
                            $display("Simulation failed");
                            $stop;
                        end
                end   
        end

endmodule