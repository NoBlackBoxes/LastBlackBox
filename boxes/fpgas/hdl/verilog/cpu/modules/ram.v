// RAM
module ram(clock, control, address, write_data, read_data);

    // Declarations
    input clock;
    input [3:0] control;
    input [31:0] address;
    input [31:0] write_data;
    output [31:0] read_data;
    
    // Intermediates (RAM)
    reg [31:0] RAM[0:4095];

    // Initialize (RAM)
    initial
        $readmemh("bin/ram.txt", RAM);

    // Intermediates (Control)
    wire write_enable = control[0];
    wire byte_enable = control[1];
    wire halfword_enable = control[2];
    wire word_enable = (~control[1] & ~control[2]);
    wire unsigned_enable = control[3];
    wire [1:0] offset = address[1:0];
    wire offset_0 = (~address[0] & ~address[1]); // Address alignment offset = 0 (00)
    wire offset_1 = (address[0] & ~address[1]);  // Address alignment offset = 1 (01)
    wire offset_2 = (~address[0] & address[1]);  // Address alignment offset = 2 (10)
    wire offset_3 = (address[0] & address[1]);   // Address alignment offset = 3 (11)
    wire [3:0] byte_flags;
    wire [7:0] byte_3, byte_2, byte_1, byte_0;
    wire [15:0] halfword_1, halfword_0;
    wire [31:0] read_word;
    wire [7:0] read_byte;
    wire [15:0] read_halfword;

    // Assign byte flag bits (bytes selected from word for memory operation)
    assign byte_flags[0] = (offset_0 & byte_enable) | (offset_0 & halfword_enable) | word_enable;
    assign byte_flags[1] = (offset_1 & byte_enable) | (offset_0 & halfword_enable) | word_enable;
    assign byte_flags[2] = (offset_2 & byte_enable) | (offset_2 & halfword_enable) | word_enable;
    assign byte_flags[3] = (offset_3 & byte_enable) | (offset_2 & halfword_enable) | word_enable;

    // Assign read word
    assign read_word = RAM[address[31:2]]; // 32-bit word aligned

    // Assign read halfwords
    assign {halfword_1, halfword_0} = read_word;

    // Assign read bytes
    assign {byte_3, byte_2, byte_1, byte_0} = read_word;

    // Assign selected byte
    assign read_byte = offset[1] ? (offset[0] ? byte_3 : byte_2) : (offset[0] ? byte_1 : byte_0);

    // Assign selected halfword
    assign read_halfword = offset[1] ? halfword_1 : halfword_0;

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