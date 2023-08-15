// RAM (NBBPU)
module ram(clock, control, address, write_data, read_data);

    // Declarations
    input clock;
    input control;
    input [15:0] address;
    input [15:0] write_data;
    output [15:0] read_data;
    
    // Intermediates (RAM)
    reg [15:0] RAM[0:255];

    // Initialize (RAM)
    initial
        $readmemh("bin/ram.txt", RAM);

    // Intermediates (Control)
    wire write_enable = control[0];
    wire [15:0] halfword_1, halfword_0;
    wire [31:0] read_word;
    wire [7:0] read_byte;
    wire [15:0] read_halfword;

    // Assign read data
    assign read_data = word_enable ? read_word : (byte_enable ? {24'h000000, read_byte} : {16'h0000, read_halfword});
    
    // Logic (write)
    always @(posedge clock)
        if (write_enable) 
            if(byte_enable)
                case(offset)
                    2'b00: RAM[address[31:2]][7:0] <= write_data[7:0];
                    2'b01: RAM[address[31:2]][15:8] <= write_data[7:0];
                    2'b10: RAM[address[31:2]][23:16] <= write_data[7:0];
                    2'b11: RAM[address[31:2]][31:24] <= write_data[7:0];
                    default: RAM[address[31:2]][7:0] <= write_data[7:0];
                endcase
            else if (halfword_enable)
                case(offset)
                    2'b00: RAM[address[31:2]][15:0] <= write_data[15:0];
                    2'b01: RAM[address[31:2]][23:8] <= write_data[15:0];
                    2'b10: RAM[address[31:2]][31:16] <= write_data[15:0];
                    default: RAM[address[31:2]][15:0] <= write_data[15:0];
                endcase
            else
                RAM[address[31:2]] <= write_data;
endmodule