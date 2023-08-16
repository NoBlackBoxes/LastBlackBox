// RAM (NBBPU)
module ram(clock, write_enable, address, write_data, read_data);

    // Declarations
    input clock;
    input write_enable;
    input [15:0] address;
    input [15:0] write_data;
    output [15:0] read_data;
    
    // Intermediates (RAM)
    reg [15:0] RAM[0:255];

    // Initialize (RAM)
    initial
        $readmemh("bin/ram.txt", RAM);

    // Assign output data
    assign read_data = RAM[address[15:1]];
    
    // Logic (write input data)
    always @(posedge clock)
        if (write_enable) 
                RAM[address[15:1]] <= write_data;
endmodule