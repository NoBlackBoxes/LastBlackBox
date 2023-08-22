// NBBSOC
// --------------------------------------------------------
// This is the top module of the NBBSOC (System-on-a-chip). 
// --------------------------------------------------------
module nbbsoc(clock, reset, RGB);
    
    // Declarations
    input clock;
    input reset;
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
    nbbpu #(.CLOCK_DIV(32'h00000001)) nbbpu(
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
    //assign RGB = ~debug_RGB;
    assign RGB = {1'b1, 1'b1, ~pulse};

    // ------------------------
    // Memory Mapped I/O (MMIO)
    // ------------------------

    // Peripherals (PWM) - Address: 16'h8000
    wire pulse;
    reg [7:0] duty_cycle;
    always @(posedge clock)
        begin
            if((write_enable == 1'b1) && (address == 16'h8000))
                duty_cycle <= write_data[7:0];
            else
                duty_cycle = sample[7:0];
        end
    pwm pwm(clock, duty_cycle, pulse);

    // Peripherals (ADC) - Address: 16'h8010
    reg adc_enable;
    wire [9:0] sample;
    always @(posedge clock)
        begin
            if((write_enable == 1'b1) && (address == 16'h8010))
                adc_enable <= 1'b1;
            else
                adc_enable <= 1'b0;
        end
    adc adc(clock, adc_enable, sample);

endmodule