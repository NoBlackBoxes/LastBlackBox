// Rx (UART)
module tx(clock, valid, byte, done, pin);

    // Parameters
    parameter BAUD_RATE = 9600;
    parameter CLOCK_HZ = 12_000_000;

    // Declarations
    input  wire clock;              // Clock Input
    input  wire valid;              // Input Valid
    input  wire [7:0] byte;         // The Byte to Transmit
    output wire done;               // Tx Done
    output reg pin;                 // Serial Transmit Pin

    // Constants
    localparam CYCLES_PER_BIT = CLOCK_HZ / BAUD_RATE;           // Clock cycles per bit
    localparam CLOCK_COUNT_SIZE = 1+$clog2(CYCLES_PER_BIT);     // Size of the register that stores clock count

    // States
    localparam FSM_IDLE = 0;        // Do nothing
    localparam FSM_START = 1;       // Start transmission
    localparam FSM_TRANSMIT = 2;    // Transmitting
    localparam FSM_STOP = 3;        // Stop transmission
    localparam FSM_CLEANUP = 4;     // Cleanup

    // Registers
    reg tx_done;                                // Tx done
    reg [2:0] state;                            // Current state
    reg [2:0] bit_index;                        // Index of current bit
    reg [7:0] internal_byte;                    // Internal storage for the transmitting byte
    reg [CLOCK_COUNT_SIZE-1:0] clock_count;     // Clock counter

    // Logic: Assign outputs 
    assign done = tx_done;     

    // Logic: State Machine
    always @(posedge clock)
        begin
            case (state)
                FSM_IDLE:
                    begin
                        pin <= 1'b1;                            // Drive Tx pin high for idle
                        tx_done <= 1'b0;
                        clock_count <= 0;
                        bit_index <= 0;                       
                        if (valid == 1'b1)                      // Transmit byte ready
                            begin
                                internal_byte <= byte;
                                state <= FSM_START;
                            end
                        else
                            state <= FSM_IDLE;
                    end
                FSM_START:
                    begin
                        pin <= 1'b0;                            // Send START bit (0)        
                        if (clock_count < CYCLES_PER_BIT-1)     // Wait CYCLES_PER_BIT-1 for start bit to transmit
                            begin
                                clock_count <= clock_count + 1;
                                state <= FSM_START;
                            end
                        else
                            begin
                                clock_count <= 0;
                                state <= FSM_TRANSMIT;
                            end
                    end                                
                FSM_TRANSMIT:
                    begin
                        pin <= internal_byte[bit_index];                        
                        if (clock_count < CYCLES_PER_BIT-1)     // Wait CYCLES_PER_BIT-1 for data bits to transmit         
                            begin
                                clock_count <= clock_count + 1;
                                state <= FSM_TRANSMIT;
                            end
                        else
                            begin
                                clock_count <= 0;                                
                                if (bit_index < 7)              // Check if we have transmitted all bits
                                    begin
                                        bit_index <= bit_index + 1;
                                        state <= FSM_TRANSMIT;
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
                        pin <= 1'b1;                            // Drive Tx pin high for STOP bit
                        if (clock_count < CYCLES_PER_BIT-1)     // Wait CYCLES_PER_BIT-1 for STOP bit to transmit
                            begin
                                clock_count <= clock_count + 1;
                                state <= FSM_STOP;
                            end
                        else
                            begin
                                tx_done <= 1'b1;
                                clock_count <= 0;
                                state <= FSM_CLEANUP;
                            end
                    end                
                FSM_CLEANUP:
                    begin
                        tx_done <= 1'b1;
                        state <= FSM_IDLE;
                    end
                
                default :
                    state <= FSM_IDLE;     
            endcase
        end
endmodule