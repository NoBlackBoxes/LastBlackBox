// Datapath (NBBPU)
module datapath(clock, reset, instruction, reg_write, PC_select, data_in, data_out, PC);

    // Declarations
    input clock;
    input reset;
    input [15:0] instruction;
    input reg_write;
    input PC_select;
    input [15:0] data_in;
    output [15:0] data_out;
    output [15:0] PC;
    
    // Intermediates
    wire [15:0] PC_next;
    wire [15:0] PC_plus2;
    wire [15:0] PC_target;
    wire [15:0] X;
    wire [15:0] Y;
    wire [15:0] Z;
    
    // Logic (select next PC)
    adder pc_add2(PC, 16'd2, PC_plus2);
    mux2 #(16) pc_mux(PC_plus2, PC_target, PC_select, PC_next);
    flopr #(16) pc_reg(clock, reset, PC_next, PC);
    
    // Logic (register file)
    regfile regfile(clock, reg_write, instruction[7:4], instruction[11:8], instruction[15:12], X, Y, Z);
    
    // Logic (ALU)
    alu alu(X, Y, instruction[3:0], Z);
    //select_read select_read(instruction[14:12], read_data, read_data_out);
    //mux8 #(32) result_mux(ALU_result, read_data_out, PC_plus4, immediate, PC_target, 32'hxxxxxxxx, 32'hxxxxxxxx, 32'hxxxxxxxx, result_select, result);

endmodule