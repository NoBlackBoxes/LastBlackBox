// Datapath (NBBPU)
module datapath(clock, reset, instruction, reg_write, jump_PC, branch_PC, read_data, address, write_data, PC);

    // Declarations
    input clock;
    input reset;
    input [15:0] instruction;
    input reg_write;
    input jump_PC;
    input branch_PC;
    input [15:0] read_data;
    output [15:0] address;
    output [15:0] write_data;
    output [15:0] PC;
    
    // Intermediates
    wire [3:0] opcode, x, y, z;
    wire [15:0] PC_next;
    wire [15:0] PC_plus1;
    wire [15:0] X;
    wire [15:0] Y;
    wire [15:0] Z;
    wire branch_or_jump_PC;

    // Assignments
    assign opcode = instruction[15:12];
    assign x = instruction[11:8];
    assign y = instruction[7:4];
    assign z = instruction[3:0];

    // Logic (register file)
    regfile regfile(clock, reg_write, z, Z, x, y, X, Y);

    // Logic (Branch)
    assign branch_or_jump_PC = (jump_PC | (Z[0] & branch_PC));

    // Logic (update PC, either PC+1 or jump target...or reset PC to 0)
    assign PC_plus1 =  PC + 16'd1;
    mux2 #(16) pc_mux(PC_plus1, X, branch_or_jump_PC, PC_next);
    flopr #(16) pc_reg(clock, reset, PC_next, PC);
    
    // Logic (ALU)
    alu alu(X, Y, instruction, read_data, PC_plus1, Z);
    
    // Set write_data and address (may not be used)
    assign address = X;
    assign write_data = Z;

endmodule