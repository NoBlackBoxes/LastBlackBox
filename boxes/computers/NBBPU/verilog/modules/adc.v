// ADC (10-bit)
module adc(clock, enable, sample);
 
    // Declarations
    input clock;
    input enable;
    output reg[9:0] sample;

    // Parameters
	//	:: 12 MHz clock / 256 = 46.875 kHZ per edge (23.44 kHz sample clock)
	// 	 : 18 sample clocks per sample = 1.3 kSamples/sec
    parameter CLOCK_DIV = 16'h00FF;

    // Intermediates
    reg sample_clock;
    reg [15:0] counter;

    // Generate sample clock
    initial 
        begin
            sample_clock <= 1'b0;
            counter <= 16'h0000;
        end
    always @(posedge clock)
        begin
			counter <= (counter + 16'h0001);
			if(counter >= CLOCK_DIV)
				begin
					counter <= 16'd0;
					sample_clock <= !sample_clock;
				end
        end

    // Intermediates
    reg D_in, D_out, bit_pos;
    reg [4:0] count;
	reg [2:0] state;
	reg [9:0] data;

    // State machine
	always @(negedge sample_clock)
		begin
			if(enable)
				begin
					D_in <= 0;
					data <= 10'b0000000000;
					sample <= 10'b0000000000;
					count <= 5'b00000; 						// c0
					state <= 3'd0;		   					// s0
				end
			else
				begin
					case(state)
						0: 
							begin
								D_in <= 1;
								count <= count + 5'b00001; 	// c1
								state <= 3'd1;   		   	// s1
							end
						1: 
							begin
								D_in <= 1;					// Single-ended
								count <= count + 5'b00001;	// c2
								state <= 3'd2;				// s2
							end
						2: 
							begin
								D_in <= 0;					// D2 (X)
								count <= count + 5'b00001;	// c3
								state <= 3'd3;				// s3
							end
						3: 
							begin
								D_in <= 0;					// D1 (ch0)
								count <= count + 5'b00001;	// c4
								state <= 3'd4;				// s4
							end
						4: 
							begin
								D_in <= 0;					// D0 (ch0)
								count <= count + 5'b00001;	// c5
								state <= 3'd5;				// s5
							end
						5: 
							begin
								if (count < 5'd8)
									begin
										D_in <= 0;					// X
										count <= count + 5'b00001;	// c6, c7
										state <= 3'd5;				// s5, s5
									end
								else
									begin
										D_in <= 0;
										bit_pos = 5'b10001 - count;
										data[bit_pos] <= D_out;
										count <= count + 5'b00001;	// c8
										state <= 3'd6;				// s6
									end
							end
						6: 
							begin
								if (count < 17)
									begin
										D_in <= 0;
										bit_pos = 5'b10001 - count;
										data[bit_pos] <= D_out;
										count <= count + 5'b00001;	// c9-16
										state <= 3'd6;				// s6
									end
								else 
									begin
										D_in <= 0;
										bit_pos = 5'b10001 - count;
										data[bit_pos] <= D_out;
										count <= count + 5'b00001;	// c17
										state <= 3'd7;				// s7
									end
							end
						7:
							begin
								sample <= data;
								D_in <= 0;
								count <= 5'b00000;
								state <= 3'd00;
							end
						default: 
							begin
								count <= 5'b00000;
								state <= 3'd00;
							end

					endcase
				end
		end

endmodule