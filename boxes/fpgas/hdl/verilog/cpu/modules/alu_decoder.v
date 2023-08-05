// ALU Decoder
module alu_decoder(opcode_b5, funct3, funct7b5, ALU_op, ALU_control);

    // Declarations
    input opcode_b5;
    input [2:0] funct3;
    input funct7b5;
    input [1:0] ALU_op;
    output reg [3:0] ALU_control;

    // Intermediates
    wire R_type_subtract;
    wire shift_arithemetic;
    assign R_type_subtract = funct7b5 & opcode_b5; // TRUE for R–type subtract   
    assign shift_arithemetic = funct7b5; // TRUE for arithmetic shift   
    
    // Logic
    always @*
        case(ALU_op)
            2'b00: ALU_control = 3'b000; // addition   
            2'b01:
                case(funct3)
                    3'b000: ALU_control = 4'b0001; // subtraction: beq
                    3'b001: ALU_control = 4'b0001; // subtraction: bne
                    3'b100: ALU_control = 4'b0101; // branch less than: slt, blt
                    3'b101: ALU_control = 4'b0101; // branch greater than: bge
                    3'b110: ALU_control = 4'b0110; // branch less than (unsigned): bltu
                    3'b111: ALU_control = 4'b0110; // branch greater than (unsigned): bgeu
                    default: ALU_control = 4'bxxxx; // ???
                endcase
            default:
                case(funct3) // R–type or I–type ALU
                    3'b000: 
                        begin
                            if (R_type_subtract)
                                ALU_control = 3'b001; // sub
                            else
                                ALU_control = 3'b000; // add, addi
                        end
                    3'b001: ALU_control = 4'b1001; // sll, slli
                    3'b010: ALU_control = 4'b0101; // slt, slti
                    3'b011: ALU_control = 4'b0110; // sltu, sltiu
                    3'b100: ALU_control = 4'b0100; // xor, xori
                    3'b101:
                        begin
                            if(shift_arithemetic)
                                ALU_control = 4'b0111; // sra, srai
                            else
                                ALU_control = 4'b1000; // srl, srli
                        end
                    3'b110: ALU_control = 4'b0011; // or, ori
                    3'b111: ALU_control = 4'b0010; // and, andi
                    default: ALU_control = 4'bxxxx; // ???
                endcase
        endcase

endmodule

// Add XOR for beq