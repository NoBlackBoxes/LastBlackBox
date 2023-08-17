// Regfile (NBBPU) - 16 x 16-bit registers - two read ports and one write port.
module regfile(clock, write_enable, write_address, write_data, read_address_1, read_address_2, read_data_1, read_data_2);

    // Declarations
    input clock;
    input write_enable;
    input [3:0] write_address;
    input [15:0] write_data;
    input [3:0] read_address_1;
    input [3:0] read_address_2;
    output [15:0] read_data_1;
    output [15:0] read_data_2;
    
    // Intermediates (16 x 16-bit registers)
    reg [15:0] registers[15:0];
    
    // Initialize (DEBUG)
    initial
        begin
            registers[0] = 16'd0;
            registers[1] = 16'd0;
            registers[2] = 16'd2;
            registers[3] = 16'd3;
            registers[4] = 16'd0;
            registers[5] = 16'd0;
            registers[6] = 16'd0;
            registers[7] = 16'd0;
            registers[8] = 16'd0;
            registers[9] = 16'd0;
            registers[10] = 16'd0;
            registers[12] = 16'd0;
            registers[13] = 16'd0;
            registers[14] = 16'd0;
            registers[15] = 16'd0;
        end   

    // Logic
    always @(posedge clock)
        begin
            if (write_enable) 
                registers[write_address] <= write_data;
        end
    assign read_data_1 = (read_address_1 != 0) ? registers[read_address_1] : 0; // Register 0 ia always 0
    assign read_data_2 = (read_address_2 != 0) ? registers[read_address_2] : 0; // Register 0 ia always 0

endmodule