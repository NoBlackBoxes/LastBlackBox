// Counter (32-bit)
`timescale 1ns / 1ps
module counter(clock, reset, left_leds, right_leds);
 
    // Declarations
    input clock;
    input reset;
    output wire[7:0] left_leds;
    output wire[7:0] right_leds;

    // Registers     
    reg[31:0] count;

    // Logic: Report
    assign left_leds = count[31:24];
    assign right_leds = count[7:0];

    // Initialize
    initial
        begin
            count <= 0;
        end

    // Logic: Count
    always @(posedge clock)
        begin
            if(reset)
                begin
                    count <= 0;
                end
            else
                begin
                    count <= count + 1;
                end
        end
endmodule