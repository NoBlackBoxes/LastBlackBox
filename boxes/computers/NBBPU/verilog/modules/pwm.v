// PWM
module pwm(clock, duty_cycle, pulse);
 
    // Declarations
    input clock;
    input [7:0] duty_cycle;
    output pulse;

    // Intermediates
    reg [7:0] counter;
    
    // Logic
    initial 
        begin
            counter <= 8'b00000000;
        end

    // Logic (PWM)
    assign pulse = (counter < duty_cycle);

    // Increment counter
    always @(posedge clock)
        begin
            counter <= (counter + 8'b00000001);
        end

endmodule