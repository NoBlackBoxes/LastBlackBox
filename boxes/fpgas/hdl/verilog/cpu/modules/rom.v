// ROM
module rom(address, read_data);

    // Declarations
    input [31:0] address;
    output [31:0] read_data;   

    // Intermediates
    reg [31:0] ROM[0:4095];

    // Initialize
    initial
        $readmemh("bin/rom.txt", ROM);

    // Logic    
    assign read_data = ROM[address[31:2]]; // 32-bit word aligned

endmodule