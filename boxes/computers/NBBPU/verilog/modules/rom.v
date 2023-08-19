// ROM
module rom(clock, select, read_enable, address, read_data);

    // Declarations
    input clock;
    input select;
    input read_enable;
    input [15:0] address;
    output reg [15:0] read_data;   

    // Intermediates
    reg [15:0] ROM[0:255];

    // Initialize
    initial
        $readmemh("bin/rom.txt", ROM);

    // Logic (read output data)
    always @(posedge clock)
        begin
            if(select & read_enable)
                read_data <= ROM[address[15:0]];
        end

endmodule