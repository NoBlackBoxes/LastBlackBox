// ROM
module rom(address, data_out);

    // Declarations
    input [15:0] address;
    output [15:0] data_out;   

    // Intermediates
    reg [15:0] ROM[0:31];

    // Initialize
    initial
        $readmemh("bin/rom.txt", ROM);

    // Logic    
    assign data_out = ROM[address[15:1]]; // 16-bit word aligned

endmodule