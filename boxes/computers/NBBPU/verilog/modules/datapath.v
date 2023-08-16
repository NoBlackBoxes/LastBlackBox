// Datapath (NBBPU)
module datapath(clock, reset, instruction, reg_write_lower, reg_write_upper, reg_set, PC_select, read_data, address, write_data, PC);

    // opcode   = instruction[15:12]
    // x        = instruction[11:8]
    // y        = instruction[7:4]
    // z        = instruction[3:0]

    // Declarations
    input clock;
    input reset;
    input [15:0] instruction;
    input reg_write_lower;
    input reg_write_upper;
    input reg_set;
    input PC_select;
    input [15:0] read_data;
    output [15:0] address;
    output [15:0] write_data;
    output [15:0] PC;
    
    // Intermediates
    wire [15:0] PC_next;
    wire [15:0] PC_plus2;
    wire [15:0] PC_target;
    wire [15:0] reg_data_out_A;
    wire [15:0] reg_data_out_B;
    wire [15:0] reg_data_in;
    wire [15:0] ALU_result;
    
    // Logic (update PC, either PC+2 or Jump target)
    adder pc_add2(PC, 16'd2, PC_plus2);
    mux2 #(16) pc_mux(PC_plus2, PC_target, PC_select, PC_next);
    flopr #(16) pc_reg(clock, reset, PC_next, PC);
    
    // Logic (ALU)
    alu alu(reg_data_out_A, reg_data_out_B, instruction[15:12], read_data, ALU_result);

    // Logic (register file)
    regfile regfile(clock, reg_write_lower, reg_write_upper, instruction[3:0], reg_data_in, instruction[11:8], instruction[7:4], reg_data_out_A, reg_data_out_B);
    
    // What should be the register input data? ALU result vs. Set -Byte- vs. Data In From memory)
    mux2 #(16) reg_mux(ALU_result, {8'b00000000, instruction[11:8], instruction[7:4]}, reg_set, reg_data_in);

    // Set write_data and address (may not be used)
    assign address = reg_data_out_A;
    assign write_data = ALU_result;

endmodule