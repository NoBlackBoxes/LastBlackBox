// Regfile
module regfile(clock, write_enable, address_read_1, address_read_2, address_write, write_data, read_data_1, read_data_2);

    // Declarations
    input clock;
    input write_enable;
    input [5:0] address_read_1;
    input [5:0] address_read_2;
    input [5:0] address_write;
    input [31:0] write_data;
    output [31:0] read_data_1;
    output [31:0] read_data_2;
    
    // Intermediates    
    reg [31:0] registers[31:0];
    
    // Logic
    always @(posedge clock)
        begin
            if (write_enable) 
                registers[address_write] <= write_data;
        end
    assign read_data_1 = (address_read_1 != 0) ? registers[address_read_1] : 0; // Register 0 ia always 0
    assign read_data_2 = (address_read_2 != 0) ? registers[address_read_2] : 0; // Register 0 ia always 0

endmodule