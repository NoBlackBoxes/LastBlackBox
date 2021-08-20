// RAM
module ram(clock, control, address, write_data, read_data);

    // Declarations
    input clock;
    input [3:0] control;
    input [31:0] address;
    input [31:0] write_data;
    output [31:0] read_data;
    
    // Intermediates
    reg [31:0] RAM[0:4095];
    wire write_enable = control[0] | control[1] | control[2] | control[3];

    // Initialize
    initial
        $readmemh("bin/ram.txt", RAM);

    // Logic (read)
    assign read_data = RAM[address[31:2]];  // 32-bit word aligned

    // Logic (write)
    always @(posedge clock)
        if (write_enable) 
            begin
                RAM[address[31:2]] <= write_data;
            end

endmodule