// Decoders (RV32I)

// Decoder (Main)
module decoder_main(opcode, result_src, mem_write, branch, alu_src, reg_write, jump, imm_src, alu_op);

    // Declarations
    input [6:0] opcode;
    output [1:0] result_src;
    output mem_write;
    output branch;
    output alu_src;
    output reg_write;
    output jump;
    output [1:0] imm_src;
    output [1:0] alu_op;
    
    // Intermediates
    logic [10:0] controls;
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
module decoder_alu();

    // Declarations
    input logic opb5,
    input logic [2:0] funct3,
    input logic funct7b5,
    input logic [1:0] ALUOp,
    output logic [2:0] ALUControl;   

    // Intermediates
    logic  RtypeSub;
    assign RtypeSub = funct7b5 & opb5; // TRUE for R–type subtract   
    
    // Logic
    always_comb      
        case(ALUOp)   
            2'b00:      ALUControl = 3'b000; // addition   
            2'b01:      ALUControl = 3'b001; // subtraction   
            default:
                case(funct3) // R–type or I–type ALU
                    3'b000: if (RtypeSub)
                                ALUControl = 3'b001; // sub
                            else
                                ALUControl = 3'b000; // add, addi
                    3'b010:     ALUControl = 3'b101; // slt, slti
                    3'b110:     ALUControl = 3'b011; // or, ori
                    3'b111:     ALUControl = 3'b010; // and, andi
                    default:    ALUControl = 3'bxxx; // ???
                endcase
        endcase

endmodule

