// Datapath (NBBPU)
module datapath(clock, reset, result_select, PC_select, ALU_select, reg_write, ALU_control, instruction, read_data, zero, PC, ALU_result, write_data);

    // Declarations
    input clock;
    input reset;
    input [2:0] result_select;
    input [1:0] PC_select;
    input ALU_select;
    input reg_write;
    input [3:0] ALU_control;
    input [15:0] instruction;
    input [15:0] read_data;
    output zero;
    output [15:0] PC;
    output [15:0] ALU_result;
    output [15:0] write_data;
    
    // Intermediates
    wire [15:0] PC_next;
    wire [15:0] PC_plus4;
    wire [15:0] PC_target;
    wire [15:0] immediate;
    wire [15:0] src_A;
    wire [15:0] src_B;
    wire [15:0] read_data_out;
    wire [15:0] result;
    
    // Logic (next PC)
    flopr #(32) pc_reg(clock, reset, PC_next, PC);
    adder pc_add4(PC, 32'd4, PC_plus4);
    adder pc_addbranch(PC, immediate, PC_target);
    mux3 #(32) pc_mux(PC_plus4, PC_target, ALU_result, PC_select, PC_next);
    
    // Logic (register file)
    regfile regfile(clock, reg_write, instruction[19:15], instruction[24:20], instruction[11:7], result, src_A, write_data);
    generate_immediate generate_immediate(instruction, immediate);
    
    // Logic (ALU)
    mux2 #(32) src_B_mux(write_data, immediate, ALU_select, src_B);
    alu alu(src_A, src_B, ALU_control, ALU_result, zero);
    select_read select_read(instruction[14:12], read_data, read_data_out);
    mux8 #(32) result_mux(ALU_result, read_data_out, PC_plus4, immediate, PC_target, 32'hxxxxxxxx, 32'hxxxxxxxx, 32'hxxxxxxxx, result_select, result);

endmodule