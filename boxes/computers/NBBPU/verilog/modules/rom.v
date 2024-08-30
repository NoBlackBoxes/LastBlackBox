// ROM (NBBPU)
module rom(clock, read_enable, address, read_data);

    // Declarations
    input clock;
    input read_enable;
    input [15:0] address;
    output reg [15:0] read_data;   

    // Intermediates
    reg [15:0] ROM[0:255];

    // Initialize
    initial
        begin
            $readmemh("bin/rom.txt", ROM);
        end
    
    // Logic (output read data)
    always @(posedge clock)
        begin
            if(read_enable)
                read_data <= ROM[address[15:0]];
        end

endmodule