// NBBPU
// -----------------------------------------
// This is the top module of the NBBPU (CPU)
// -----------------------------------------
module nbbpu(
                fast_clock,
                reset,
                instruction, 
                read_data, 
                instruction_enable, 
                read_enable, 
                write_enable, 
                address, 
                write_data, 
                PC, 
                debug_RGB
            );
    
    // Declarations
    input fast_clock;
    input reset;
    input [15:0] instruction;
    input [15:0] read_data;
    output instruction_enable;
    output read_enable;
    output write_enable;
    output [15:0] address;
    output [15:0] write_data;
    output [15:0] PC;
    output reg [2:0] debug_RGB;

    // Parameters (Cycle States)
    localparam FETCH     = 2'b00;    // Fetch next instruction from ROM
    localparam DECODE    = 2'b01;    // Decode instruction and generate control signals
    localparam EXECUTE   = 2'b10;    // Execute instruction inside ALU
    localparam STORE     = 2'b11;    // Store results in memory (register file or RAM)

    // Intermediates
    reg [1:0] current_state;
    reg [1:0] next_state;
    wire [3:0] opcode, _x, x, y, z;
    wire reg_write;
    wire reg_set;
    wire jump_PC;
    wire branch_PC;
    wire store_PC;
    wire branch_or_jump_PC;
    wire [15:0] X;
    wire [15:0] Y;
    wire [15:0] Z;
    wire [15:0] PC_next;
    wire [15:0] PC_plus1;

    // Assignments
    assign opcode = instruction[15:12];
    assign _x = instruction[11:8];
    assign y = instruction[7:4];
    assign z = instruction[3:0];
    assign address = X;
    assign write_data = Z;

    // Sub-module: Controller
    controller controller
    (
        current_state,          // (input) Cycle state
        opcode,                 // (input) Op Code
        instruction_enable,     // (output) Instruction read enable
        read_enable,            // (output) Memory read enable
        write_enable,           // (output) Memory write enable
        reg_write,              // (output) Register write
        reg_set,                // (output) Register set
        jump_PC,                // (output) jump PC signal
        branch_PC,              // (output) branch PC signal
        store_PC                // (output) store PC signal
    );

    // Registers
    flopenr #(16) pc_reg(clock, reset, store_PC, PC_next, PC);

    // Muxes
    mux2 #(16) pc_next_mux(PC_plus1, X, branch_or_jump_PC, PC_next);

    // Logic (PC)
    assign PC_plus1 = PC + 1;

    // Logic (register set)
    mux2 #(4) x_mux(_x, z, reg_set, x);
    
    // Logic (branch) - !!! This is wrong, all of Z should be zero...I think...not just Z[0]
    assign branch_or_jump_PC = (jump_PC || (Z[0] && branch_PC));

    // Register File
    regfile regfile(clock, reg_write, z, Z, x, y, X, Y);

    // ALU
    alu alu(X, Y, instruction, read_data, PC, Z);

    // Define State Machine
    initial 
        begin
           current_state = FETCH;
        end
    always @(*)
        begin
            case(current_state)
                FETCH:      next_state = DECODE;
                DECODE:     next_state = EXECUTE;
                EXECUTE:    next_state = STORE;
                STORE:      next_state = FETCH;
            endcase
        end

    // Update State
    always @(posedge clock, posedge reset)
            if(reset)       current_state <= FETCH;
            else            current_state <= next_state;

    // --------------------------

    // DEBUG: Parameters
    reg clock;                              // Slow clock
    reg [31:0] counter;                     // Counter to produce slow clock
    parameter CLOCK_DIV = 32'h000FFFFF;

    // DEBUG: Assign RGB output
    wire pass;
    reg passed;
    assign pass = (address == 16'hFFF0) && (write_data == 16'd42);

    // DEBUG: Generate Slow Clock
    always @(posedge fast_clock, posedge reset)
        begin
            if(reset)
                begin
                    passed <= 1'b0;
                    counter <= 32'd0;
                    clock <= 1'b0;
                end
            else
                begin
                    counter <= counter + 32'd1;
                    if(counter >= CLOCK_DIV)
                        begin
                            counter <= 32'd0;
                            clock <= !clock;
                        end
                    if(pass && !passed)
                        passed <= 1'b1;
                    if (passed)
                        debug_RGB <= {1'b0, 1'b1, 1'b0};
                    else
                        begin
                            if(write_enable)
                                debug_RGB <= write_data[2:0];
                        end
                end
        end
endmodule