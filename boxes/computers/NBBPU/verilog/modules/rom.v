// ROM
module rom(clock, select, address, data_out);

    // Declarations
    input clock;
    input select;
    input [15:0] address;
    output reg [15:0] data_out;   

    // Intermediates
    reg [15:0] ROM[0:255];

    // Initialize
    initial
        $readmemh("bin/rom.txt", ROM);

    // Logic    
    always @(posedge clock)
        begin
            if(select)
                data_out <= ROM[address[15:0]];
        end

endmodule