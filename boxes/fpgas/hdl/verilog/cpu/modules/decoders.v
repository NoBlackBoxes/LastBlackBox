// Decoders (RV32I)

// Decoder (Main)
module decoder_main(opcode, result_src, mem_write, branch, ALU_src, reg_write, jump, imm_src, alu_op);

    // Declarations
    input [6:0] opcode;
    output [1:0] result_src;
    output mem_write;
    output branch;
    output ALU_src;
    output reg_write;
    output jump;
    output [1:0] imm_src;
    output [1:0] ALU_op;
    
    // Intermediates
    wire [10:0] controls;
    assign {reg_write, imm_src, alu_src, mem_write, result_src, branch, alu_op, jump} = controls;   
    
    // Logic
    always_comb
        case(op)      
            // reg_write - imm_src - alu_src - mem_write - result_src - branch - alu_op - jump
            7'b0000011: controls = 11'b1_00_1_0_01_0_00_0; // lw
            7'b0100011: controls = 11'b0_01_1_1_00_0_00_0; // sw
            7'b0110011: controls = 11'b1_xx_0_0_00_0_10_0; // R–type
            7'b1100011: controls = 11'b0_10_0_0_00_1_01_0; // beq
            7'b0010011: controls = 11'b1_00_1_0_00_0_10_0; // I–type ALU
            7'b1101111: controls = 11'b1_11_0_0_10_0_00_1; // jal
            default:    controls = 11'bx_xx_x_x_xx_x_xx_x; // ???   
        endcase
endmodule

// Decoder (ALU)
module decoder_alu(opcode_b5, funct3, funct7b5, ALU_op, ALU_control);

    // Declarations
    input opcode_b5,
    input [2:0] funct3,
    input funct7b5,
    input [1:0] ALU_op,
    output [2:0] ALU_control;   

    // Intermediates
    wire  Rtype_sub;
    assign Rtype_sub = funct7b5 & opb5; // TRUE for R–type subtract   
    
    // Logic
    always_comb      
        case(ALU_op)   
            2'b00:      ALU_control = 3'b000; // addition   
            2'b01:      ALU_control = 3'b001; // subtraction   
            default:
                case(funct3) // R–type or I–type ALU
                    3'b000: if (Rtype_sub)
                                ALU_control = 3'b001; // sub
                            else
                                ALU_control = 3'b000; // add, addi
                    3'b010:     ALU_control = 3'b101; // slt, slti
                    3'b110:     ALU_control = 3'b011; // or, ori
                    3'b111:     ALU_control = 3'b010; // and, andi
                    default:    ALU_control = 3'bxxx; // ???
                endcase
        endcase

endmodule

