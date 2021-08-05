// Datapath (RV32I)
module datapath(clock, reset, result_select, PC_select, ALU_select, reg_write, immediate_select, ALU_control, instruction, read_data, zero, PC, ALU_result, write_data);

    // Declarations
    input clock;
    input reset;
    input [1:0] result_select;
    input PC_select;
    input ALU_select;
    input reg_write;
    input [1:0] immediate_select;
    input [2:0] ALU_control;
    input [31:0] instruction;
    input [31:0] read_data;
    output zero;
    output [31:0] PC;
    output [31:0] ALU_result;
    output [31:0] write_data;
    
    // Intermediates
    wire [31:0] PC_next;
    wire [31:0] PC_plus4;
    wire [31:0] PC_target;   
    wire [31:0] immediate_extended;   
    wire [31:0] src_A;
    wire [31:0] src_B;   
    wire [31:0] result;   
    
    // Logic (next PC)   
    flopr #(32) pc_reg(clock, reset, PC_next, PC);
    adder pc_add4(PC, 32'd4, PC_plus4);   
    adder pc_addbranch(PC, immediate_extended, PC_target);   
    mux2 #(32) pcmux(PC_plus4, PC_target, PC_select, PC_next);   
    
    // Logic (register file)
    regfile regfile(clock, reg_write, instruction[19:15], instruction[24:20], instruction[11:7], result, src_A, write_data);
    extend extend(instruction, immediate_select, immediate_extended);   
    
    // Logic (ALU)
    mux2 #(32) src_B_mux(write_data, immediate_extended, ALU_select, src_B);
    alu alu(src_A, src_B, ALU_control, ALU_result, zero);
    mux3 #(32) result_mux(ALU_result, read_data, PC_plus4, result_select, result);

endmodule