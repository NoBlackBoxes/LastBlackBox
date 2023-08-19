// RAM (NBBPU)
module ram(clock, select, write_enable, address, write_data, read_data);

    // Declarations
    input clock;
    input select;
    input write_enable;
    input [15:0] address;
    input [15:0] write_data;
    output reg [15:0] read_data;
    
    // Intermediates (RAM)
    reg [15:0] RAM[0:255];

    // Initialize (RAM)
    initial
        $readmemh("bin/ram.txt", RAM);

    // Logic (read output data)
    always @(posedge clock)
        begin
            if(select)
                read_data <= RAM[address[15:0]];
        end

    // Logic (write input data)
    always @(posedge clock)
        begin
            if (select & write_enable) 
                RAM[address[15:0]] <= write_data;
        end
        
endmodule