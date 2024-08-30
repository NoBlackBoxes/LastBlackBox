// ADC (10-bit)
module adc(clock, D_out, CSn, sample_clock, D_in, tx_pin, leds, gpio);
 
    // Declarations
    input  wire clock;			// System Clock
	input  wire D_out;			// Serialized Sample Bit (from ADC)
	output reg 	sample_clock;	// ADC Sample Clock
	output reg  CSn;			// ADC Chip Select/Shutdown (active low)
	output reg  D_in;			// Serialized Control Bit (to ADC)
	output wire tx_pin;			// UART Tx pin
    output wire [7:0] leds; 	// Reporter LEDs
    output wire [5:0] gpio; 	// Debug GPIOs

    // Parameters
	//	:: 12 MHz clock / 256 = 46.875 kHZ per edge (23.44 kHz sample clock)
	// 	 : 18 sample clocks per sample = 1.3 kSamples/sec
    parameter SAMPLE_CLOCK_DIV = 24'h0000FF;
    parameter CLOCK_HZ = 12_000_000;
    parameter BAUD_RATE = 115200;

    // Intermediates
	reg  tx_valid;				// Tx sample is valid (start transmit)
	wire tx_done;				// Tx done (transmit finished)
	reg  tx_ready;				// Tx ready to transmit
	reg  [4:0] state;			// ADC state
    reg  [7:0] tx_byte;			// Tx data byte
	reg  [9:0] data;			// Data register
    reg  [23:0] counter;		// Sample clock counter

    // Logic: Report
    assign gpio[4:0] = state;
    assign leds = tx_byte;

    // UART Transmitter (Tx) module
    tx #(.BAUD_RATE(BAUD_RATE), .CLOCK_HZ(CLOCK_HZ)) uart_tx(
        .clock (clock),
        .valid (tx_valid),
        .byte  (tx_byte),
        .done  (tx_done),
        .pin   (tx_pin) 
    );

    // Initialize
    initial
        begin
            sample_clock <= 1'b0;
			tx_valid <= 1'b0;
			tx_ready <= 1'b0;
			state <= 5'd0;
			tx_byte <= 8'b00000000;
			data <= 24'd0;
            counter <= 24'h00000;
			CSn <= 1'b1;
        end

	// Generate transmit pulse
	always @(posedge tx_ready or posedge tx_done)
		begin
			if(tx_done)
				tx_valid <= 1'b0;
			else
				tx_valid <= 1'b1;
		end

    // Generate sample clock
    always @(posedge clock)
        begin
			counter <= (counter + 24'h00001);
			if(counter >= SAMPLE_CLOCK_DIV)
				begin
					counter <= 24'd0;
					sample_clock <= !sample_clock;
				end
		end

    // State machine
	always @(negedge sample_clock)
		begin
			case(state)
				0: 
					begin
						state <= 5'd1;   		   	// s1
						D_in <= 0;
						CSn <= 1'b1;
					end
				1: 
					begin
						state <= 5'd2;   		   	// s1
						CSn <= 1'b0;
					end
				2: 
					begin
						state <= 5'd3;   		   	// s1
					end
				3: 
					begin
						D_in <= 1;
						state <= 5'd4;   		   	// s1
						tx_ready <= 1'b0;			// UART Tx not ready to transmit
					end
				4: 
					begin
						D_in <= 1;					// Single-ended
						state <= 5'd5;				// s2
					end
				5: 
					begin
						D_in <= 0;					// D2 (X)
						state <= 5'd6;				// s3
					end
				6: 
					begin
						D_in <= 0;					// D1 (ch0)
						state <= 5'd7;				// s4
					end
				7: 
					begin
						D_in <= 0;					// D0 (ch0)
						state <= 5'd8;				// s5
					end
				8: 
					begin
						D_in <= 0;					// X
						state <= 5'd9;				// s6
					end
				9: 
					begin
						D_in <= 0;					// X
						state <= 5'd10;				// s7
					end
				10:
					begin
						D_in <= 0;
						data[9] <= D_out;			// b9
						state <= 5'd11;				// s8
					end
				11:
					begin
						D_in <= 0;
						data[8] <= D_out;			// b8
						state <= 5'd12;				// s9
					end
				12:
					begin
						D_in <= 0;
						data[7] <= D_out;			// b7
						state <= 5'd13;				// s10
					end
				13:
					begin
						D_in <= 0;
						data[6] <= D_out;			// b6
						state <= 5'd14;				// s11
					end
				14:
					begin
						D_in <= 0;
						data[5] <= D_out;			// b5
						state <= 5'd15;				// s12
					end
				15:
					begin
						D_in <= 0;
						data[4] <= D_out;			// b4
						state <= 5'd16;				// s13
					end
				16:
					begin
						D_in <= 0;
						data[3] <= D_out;			// b3
						state <= 5'd17;				// s14
					end
				17:
					begin
						D_in <= 0;
						data[2] <= D_out;			// b2
						state <= 5'd18;				// s15
					end
				18:
					begin
						D_in <= 0;
						data[1] <= D_out;			// b1
						state <= 5'd19;				// s16
					end
				19:
					begin
						D_in <= 0;
						data[0] <= D_out;			// b0
						state <= 5'd20;				// s17
					end
				20:
					begin
						D_in <= 0;
						state <= 5'd0;				// s0
						tx_ready <= 1'b1;			// UART Tx ready to transmit
						tx_byte <= data[9:2];		// Only most significant 8-bits get transmitted
						CSn <= 1'b1;
					end
				default: 
					begin
						state <= 5'd0;				// s0
					end
			endcase
		end

endmodule