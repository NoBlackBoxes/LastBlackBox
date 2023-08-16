// Datapath (NBBPU)
module datapath(clock, reset, instruction, reg_write_lower, reg_write_upper, reg_set, PC_select, read_data, write_data, PC);

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
    //select_read select_read(instruction[14:12], read_data, read_write_data);
    //mux8 #(32) result_mux(ALU_result, read_write_data, PC_plus4, immediate, PC_target, 32'hxxxxxxxx, 32'hxxxxxxxx, 32'hxxxxxxxx, result_select, result);

    // Logic (register file)
    regfile regfile(clock, reg_write_lower, reg_write_upper, instruction[3:0], reg_data_in, instruction[11:8], instruction[7:4], reg_data_out_A, reg_data_out_B);
    
    // What is register input data? ALU result vs. Set -Byte- vs. Data In From memory)
    mux2 #(16) reg_mux(ALU_result, {8'b00000000, instruction[11:8], instruction[7:4]}, reg_set, reg_data_in);

endmodule