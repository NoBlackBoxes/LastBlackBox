// NBBPU
// -----------------------------------------
// This is the top module of the NBBPU (CPU)
// -----------------------------------------
module nbbpu(
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
                debug
            );
    
    // Declarations
    input clock;
    input reset;
    input [15:0] instruction;
    input [15:0] read_data;
    output instruction_enable;
    output read_enable;
    output write_enable;
    output [15:0] address;
    output [15:0] write_data;
    output reg [15:0] PC;
    output reg debug;

    // Parameters (Cycle States)
    parameter FETCH     = 2'b00;    // Fetch next instruction from ROM
    parameter DECODE    = 2'b01;    // Decode instruction and generate control signals
    parameter EXECUTE   = 2'b10;    // Execute instruction inside ALU
    parameter STORE     = 2'b11;    // Store results in memory (register file or RAM)

    // Intermediates
    reg [1:0] current_state;
    reg [1:0] next_state;
    wire [3:0] opcode, _x, x, y, z;
    wire reg_write;
    wire reg_set;
    wire jump_PC;
    wire branch_PC;
    wire branch_or_jump_PC;
    wire [15:0] X;
    wire [15:0] Y;
    wire [15:0] Z;
    reg [15:0] PC_next;

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
        read_enable,            // (output) Data read enable
        reg_write,              // (output) Register write enable
        reg_set,                // (output) Register set enable
        write_enable,           // (output) Data write enable
        jump_PC,                // (output) jump PC signal
        branch_PC               // (output) branch PC signal
    );

    // Logic (register set)
    mux2 #(4) x_mux(_x, z, reg_set, x);

    // Logic (register file)
    regfile regfile(clock, reg_write, z, Z, x, y, X, Y);

    // Logic (branch)
    assign branch_or_jump_PC = (jump_PC | (Z[0] & branch_PC));

    // Logic (ALU)
    alu alu(X, Y, instruction, read_data, PC, Z);

    // Cycle State Machine
    initial 
        begin
            current_state = FETCH;
            PC = 0;
        end
    always @(*)
        begin
            case(current_state)
                FETCH:
                    begin
                        next_state = DECODE;
                    end
                DECODE:
                    begin
                        next_state = EXECUTE;
                    end
                EXECUTE:
                    begin
                        if(branch_or_jump_PC)
                            PC_next = X;
                        else
                            PC_next = PC + 1;
                        next_state = STORE;
                    end
                STORE:
                    begin
                        PC = PC_next;
                        next_state = FETCH;
                    end
            endcase
        end

    // Update State
    always @(posedge clock)
        begin
            if(!reset) 
                begin
                    current_state <= FETCH;
                    PC <= 0;
                end
            else
                begin
                    current_state <= next_state;
                end
        end

    // Assign Debug Signal
    initial 
        begin
            debug = 1'b1;
        end
    always @(posedge clock)
        begin
            if((write_data[15] == 1'b1) & write_enable)
                debug = write_data[0];     
        end

endmodule