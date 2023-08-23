// NBBSOC
// --------------------------------------------------------
// This is the top module of the NBBSOC (System-on-a-chip). 
// --------------------------------------------------------
module nbbsoc(clock, reset, ADC_data, ADC_control, RGB);
    
    // Declarations
    input clock;
    input reset;
    input ADC_data;
    output [2:0] ADC_control;
    output [2:0] RGB;

    // Intermediates
    wire select;
    wire [15:0] instruction;
    wire [15:0] read_data;
    wire instruction_enable;
    wire read_enable;
    wire write_enable;
    wire [15:0] address;
    wire [15:0] write_data;
    wire [15:0] PC;
    wire [2:0] debug_RGB;

    // Assignments
    assign select = 1'b1;

    // CPU module
    nbbpu #(.CLOCK_DIV(32'h00000FFF)) nbbpu(
                clock, 
                reset, 
                instruction, 
                read_data, 
                instruction_enable, 
                read_enable, 
                write_enable, 
                address, 
                write_data, 
                PC, 
                debug_RGB);
    
    // Create Instruction and Data Memory modules
    rom rom(clock, instruction_enable, PC, instruction);
    ram ram(clock, read_enable, write_enable, address, write_data, read_data);
    
    // Assign Debug signals
    assign RGB = {~pulse, 1'b1, ~pulse};

    // ------------------------
    // Memory Mapped I/O (MMIO)
    // ------------------------

    // Peripherals (PWM) - Address: 16'h8000
    wire pulse;
    reg [7:0] duty_cycle;
    pwm pwm(clock, duty_cycle, pulse);

    // Peripherals (ADC) - Address: 16'h8010
    reg adc_reset;
    wire sample_clock;
    wire D_in;
    wire [4:0] adc_state;
    wire [9:0] sample;
    adc adc(clock, adc_reset, ADC_data, sample_clock, D_in, adc_state, sample);
    assign ADC_control = {adc_reset, sample_clock, D_in};

    // Peripherals (MMIO)
    always @(posedge clock)
        begin
            if(write_enable == 1'b1)
                begin 
                    case (address)
                        16'h8000:
                            duty_cycle <= write_data[7:0];
                        16'h8010:
                            adc_reset <= 1'b1;
                        default:
                            adc_reset <= 1'b0;
                    endcase
                end
            else if(read_enable == 1'b1)
                begin 
                    case (address)      // Need to deal with bus conflict!
                        16'h8010:
                            read_data <= {6'b000000, sample};
                        default:
                            adc_reset <= 1'b0;
                    endcase
                end
            else
                begin
                    adc_reset <= 1'b0;
                    duty_cycle = sample[9:2];
                end
        end

    // --------------------
    // Communication (UART) 
    // --------------------


endmodule