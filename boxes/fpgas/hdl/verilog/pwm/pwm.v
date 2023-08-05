// PWM
`timescale 1ns / 1ps
module pwm(clock, duty_cycle, pulse);
 
    // Declarations
    input clock;
    input [7:0] duty_cycle;
    output reg pulse;

    // Intermediates
    reg [7:0] count;
    
    // Logic
    initial 
        begin
            count <= 8'b00000000;
            pulse <= 1'b0;
        end

    // Reset counter on duty_cycle change
    always @(duty_cycle)
        begin
            count <= 0;
        end

    always @(posedge clock)
        begin
            if(count == duty_cycle)
                begin
                    count <= 0;
                    pulse <= ~pulse;
                end
            else
                begin
                    count <= count + 1;
                end
        end

endmodule