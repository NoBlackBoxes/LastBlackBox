// RAM (NBBPU)
module ram(clock, read_enable, write_enable, address, write_data, read_data);

    // Declarations
    input clock;
    input read_enable;
    input write_enable;
    input [15:0] address;
    input [15:0] write_data;
    output reg [15:0] read_data;
    
    // Intermediates (RAM)
    reg [15:0] RAM[0:255];

    // Initialize (RAM)
    initial
        begin
            $readmemh("bin/ram.txt", RAM);
        end
    
    // Logic (output read data)
    always @(posedge clock)
        begin
            if(read_enable)
                read_data <= RAM[address[15:0]];
        end

    // Logic (input write data)
    always @(posedge clock)
        begin
            if (write_enable) 
                RAM[address[15:0]] <= write_data;
        end
        
endmodule