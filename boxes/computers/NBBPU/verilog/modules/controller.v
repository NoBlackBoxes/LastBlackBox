// Controller (NBBPU)
// ------------------------------------------------------------------------
// This is the "controller" sub-module for the NBBPU. It is responsible for
// decoding the opcode and generating the required control signals. 
// ------------------------------------------------------------------------
module controller(
                    state,
                    opcode,
                    instruction_enable,
                    read_enable,
                    write_enable,
                    reg_write,
                    reg_set,
                    jump_PC,
                    branch_PC,
                    store_PC
                );

    // Declarations
    input [1:0] state;
    input [3:0] opcode;
    output instruction_enable;
    output read_enable;
    output write_enable;
    output reg_write;
    output reg_set;
    output jump_PC;
    output branch_PC;
    output store_PC;

    // Parameters (Op Codes)
    localparam ADD = 4'b0000;
    localparam SUB = 4'b0001;
    localparam AND = 4'b0010;
    localparam IOR = 4'b0011;
    localparam XOR = 4'b0100;
    localparam SHR = 4'b0101;
    localparam SHL = 4'b0110;
    localparam CMP = 4'b0111;
    localparam JMP = 4'b1000;
    localparam BRZ = 4'b1001;
    localparam BRN = 4'b1010;
    localparam RES = 4'b1011;
    localparam LOD = 4'b1100;
    localparam STR = 4'b1101;
    localparam SEL = 4'b1110;
    localparam SEU = 4'b1111;

    // Parameters (Cycle States)
    localparam FETCH     = 2'b00;    // Fetch next instruction from ROM
    localparam DECODE    = 2'b01;    // Decode instruction and generate control signals
    localparam EXECUTE   = 2'b10;    // Execute instruction inside ALU
    localparam STORE     = 2'b11;    // Store results in memory (register file or RAM)

    // Logic (controls)
    assign instruction_enable = (state == FETCH);
    assign read_enable = (opcode == LOD) && (state == EXECUTE);
    assign write_enable = (opcode == STR) && (state == STORE);
    assign reg_write = ((!opcode[3]) || (opcode == JMP) || (opcode == LOD) || reg_set) && (state == STORE);
    assign reg_set = (opcode[3:1] == 3'b111);
    assign jump_PC = opcode == JMP;
    assign branch_PC = (opcode == BRZ) || (opcode == BRN);
    assign store_PC = (state == STORE);
    
endmodule