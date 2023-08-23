// ADC (10-bit)
module adc(clock, reset, D_out, sample_clock, D_in, state, sample);
 
    // Declarations
    input clock;
    input reset;
	input D_out;
	output reg sample_clock;
	output reg D_in;
	output reg [4:0] state;
    output reg[9:0] sample;

    // Parameters
	//	:: 12 MHz clock / 256 = 46.875 kHZ per edge (23.44 kHz sample clock)
	// 	 : 18 sample clocks per sample = 1.3 kSamples/sec
    parameter CLOCK_DIV = 24'h000FF;

    // Intermediates
    reg [23:0] counter;

    // Generate sample clock
    initial 
        begin
            sample_clock <= 1'b0;
            counter <= 24'h00000;
        end
    always @(posedge clock)
        begin
			counter <= (counter + 24'h00001);
			if(counter >= CLOCK_DIV)
				begin
					counter <= 24'd0;
					sample_clock <= !sample_clock;
				end
        end

    // Intermediates
	reg new;
	reg [9:0] data;

    // State machine
	always @(negedge sample_clock)
		begin
			if(reset)
				begin
					new = 1'b1;
					D_in = 0;
					data <= 10'd0;
					state <= 5'd0;		   					// s0
				end
			else
				begin
					case(state)
						0: 
							begin
								D_in = 1;
								state <= 5'd1;   		   	// s1
							end
						1: 
							begin
								D_in = 1;					// Single-ended
								state <= 5'd2;				// s2
							end
						2: 
							begin
								D_in = 0;					// D2 (X)
								state <= 5'd3;				// s3
							end
						3: 
							begin
								D_in = 0;					// D1 (ch0)
								state <= 5'd4;				// s4
							end
						4: 
							begin
								D_in = 0;					// D0 (ch0)
								state <= 5'd5;				// s5
							end
						5: 
							begin
								D_in = 0;					// X
								state <= 5'd6;				// s6
							end
						6: 
							begin
								D_in = 0;					// X
								state <= 5'd7;				// s7
							end
						7:
							begin
								D_in = 0;
								data[9] = D_out;			// b9
								state <= 5'd8;				// s8
							end
						8:
							begin
								D_in = 0;
								data[8] = D_out;			// b8
								state <= 5'd9;				// s9
							end
						9:
							begin
								D_in = 0;
								data[7] = D_out;			// b7
								state <= 5'd10;				// s10
							end
						10:
							begin
								D_in = 0;
								data[6] = D_out;			// b6
								state <= 5'd11;				// s11
							end
						11:
							begin
								D_in = 0;
								data[5] = D_out;			// b5
								state <= 5'd12;				// s12
							end
						12:
							begin
								D_in = 0;
								data[4] = D_out;			// b4
								state <= 5'd13;				// s13
							end
						13:
							begin
								D_in = 0;
								data[3] = D_out;			// b3
								state <= 5'd14;				// s14
							end
						14:
							begin
								D_in = 0;
								data[2] = D_out;			// b2
								state <= 5'd15;				// s15
							end
						15:
							begin
								D_in = 0;
								data[1] = D_out;			// b1
								state <= 5'd16;				// s16
							end
						16:
							begin
								D_in = 0;
								data[0] = D_out;			// b0
								state <= 5'd17;				// s17
							end
						17:
							begin
								D_in = 0;
								state <= 5'd0;				// s0
								if(new)
									begin
										new = 1'b0;
										sample <= data;
									end
							end
						default: 
							begin
								state <= 5'd0;
							end
					endcase
				end
		end

endmodule