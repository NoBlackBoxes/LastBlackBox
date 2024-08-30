// Rx (UART)
module rx(clock, pin, valid, byte);

    // Parameters
    parameter BAUD_RATE = 9600;
    parameter CLOCK_HZ = 12_000_000;

    // Declarations
    input  wire clock;          // Clock Input
    input  wire pin;            // Serial Recieve Pin
    output wire valid;          // Output Valid
    output wire [7:0] byte;     // The Recieved Byte

    // Constants
    localparam CYCLES_PER_BIT = CLOCK_HZ / BAUD_RATE;           // Clock cycles per bit
    localparam CLOCK_COUNT_SIZE = 1+$clog2(CYCLES_PER_BIT);     // Size of the register that stores clock count

    // States
    localparam FSM_IDLE = 0;    // Do nothing
    localparam FSM_START = 1;   // Start reception
    localparam FSM_RECEIVE = 2; // Still receiving
    localparam FSM_STOP = 3;    // Stop reception
    localparam FSM_CLEANUP = 4; // Cleanup

    // Registers
    reg serial_data;                            // Serial Rx data
    reg serial_data_buffer;                     // Double-registered serial Rx data
    reg data_valid;                             // Data valid flag
    reg [2:0] state;                            // Current state
    reg [2:0] bit_index;                        // Index of current bit
    reg [7:0] internal_byte;                    // Internal storage for the received byte
    reg [CLOCK_COUNT_SIZE-1:0] clock_count;     // Clock counter

    // Output assignment
    assign valid = data_valid;
    assign byte = internal_byte;
  
    // Logic: Double-register serial Rx pin data
    always @(posedge clock)
        begin
            serial_data_buffer <= pin;
            serial_data <= serial_data_buffer;
        end

    // Logic: State Machine
    always @(posedge clock)
        begin
            case (state)
                FSM_IDLE:
                    begin
                        data_valid <= 1'b0;
                        clock_count <= 0;
                        bit_index <= 0;
                        if (serial_data == 1'b0)  // Start bit detected
                            begin
                                state <= FSM_START;
                            end
                        else
                            state <= FSM_IDLE;
                    end
                FSM_START:
                    begin
                        if (clock_count == (CYCLES_PER_BIT-1)/2) // Check middle of start bit, is it still low?
                            begin
                                if (serial_data == 1'b0)
                                    begin
                                        clock_count <= 0;  // Reset counter, start receiving
                                        state <= FSM_RECEIVE;
                                    end
                                else
                                    state <= FSM_IDLE;
                            end
                        else
                            begin
                                clock_count <= clock_count + 1;
                                state <= FSM_START;
                            end
                    end                    
                FSM_RECEIVE:
                    begin
                        if (clock_count < CYCLES_PER_BIT-1) // Wait CYCLES_PER_BIT-1, then sample serial data
                            begin
                                clock_count <= clock_count + 1;
                                state <= FSM_RECEIVE;
                            end
                        else
                            begin
                                clock_count <= 0;
                                internal_byte[bit_index] <= serial_data;
                                if (bit_index < 7) // Check if all bits received
                                    begin
                                        bit_index <= bit_index + 1;
                                        state <= FSM_RECEIVE;
                                    end
                                else
                                    begin
                                        bit_index <= 0;
                                        state <= FSM_STOP;
                                    end
                            end
                    end            
                FSM_STOP:
                    begin
                        if (clock_count < CYCLES_PER_BIT-1) // Wait CYCLES_PER_BIT-1 for STOP bit to finish
                            begin
                                clock_count <= clock_count + 1;
                                state <= FSM_STOP;
                            end
                        else
                            begin
                                data_valid <= 1'b1;
                                clock_count <= 0;
                                state <= FSM_CLEANUP;
                            end
                    end
                FSM_CLEANUP:
                    begin
                        state <= FSM_IDLE;
                        data_valid <= 1'b0;
                    end                    
                
                default:
                    state <= FSM_IDLE;                    
            endcase
        end
endmodule