module rgb_blink (
  // outputs
  output wire led_red  , // Red
  output wire led_blue , // Blue
  output wire led_green, // Green
  output wire[7:0] left_leds,
  output wire[7:0] right_leds
);

  wire        int_osc            ;
  reg  [27:0] frequency_counter_i;
  assign left_leds = 0;
  assign right_leds = 0;

// Oscilator
SB_HFOSC u_SB_HFOSC (.CLKHFPU(1'b1), .CLKHFEN(1'b1), .CLKHF(int_osc));

// Counter
always @(posedge int_osc) begin
  frequency_counter_i <= frequency_counter_i + 1'b1;
end

// Colours and stuff
  SB_RGBA_DRV RGB_DRIVER (
    .RGBLEDEN(1'b1                                            ),
    .RGB0PWM (frequency_counter_i[25]&frequency_counter_i[24] ),
    .RGB1PWM (frequency_counter_i[25]&~frequency_counter_i[24]),
    .RGB2PWM (~frequency_counter_i[25]&frequency_counter_i[24]),
    .CURREN  (1'b1                                            ),
    .RGB0    (led_green                                       ), //Actual Hardware connection
    .RGB1    (led_blue                                        ),
    .RGB2    (led_red                                         )
  );
  defparam RGB_DRIVER.RGB0_CURRENT = "0b000001";
  defparam RGB_DRIVER.RGB1_CURRENT = "0b000001";
  defparam RGB_DRIVER.RGB2_CURRENT = "0b000001";

endmodule