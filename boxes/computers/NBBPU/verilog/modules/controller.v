// Controller (NBBPU)
// -----------------------------------------
// This is the "controller" sub-module for the NBBPU. It is responsible for decoding the opcode and generating the required
// control signals. 
// -----------------------------------------
module controller(
                    state, 
                    opcode, 
                    instruction_enable, 
                    read_enable, 
                    reg_write, 
                    reg_set, 
                    write_enable, 
                    jump_PC, 
                    branch_PC
                );

    // Declarations
    input [1:0] state;
    input [3:0] opcode;
    output instruction_enable;
    output read_enable;
    output reg_write;
    output reg_set;
    output write_enable;
    output jump_PC;
    output branch_PC;

    // Parameters (Op Codes)
    parameter ADD = 4'b0000;
    parameter SUB = 4'b0001;
    parameter AND = 4'b0010;
    parameter IOR = 4'b0011;
    parameter XOR = 4'b0100;
    parameter SHR = 4'b0101;
    parameter SHL = 4'b0110;
    parameter CMP = 4'b0111;
    parameter JMP = 4'b1000;
    parameter BRZ = 4'b1001;
    parameter BRN = 4'b1010;
    parameter RES = 4'b1011;
    parameter LOD = 4'b1100;
    parameter STR = 4'b1101;
    parameter SEL = 4'b1110;
    parameter SEU = 4'b1111;

    // Parameters (Cycle States)
    parameter FETCH     = 2'b00;    // Fetch next instruction from ROM
    parameter DECODE    = 2'b01;    // Decode instruction and generate control signals
    parameter EXECUTE   = 2'b10;    // Execute instruction inside ALU
    parameter STORE     = 2'b11;    // Store results in memory (register file or RAM)

    // Intermediates
    reg [6:0] controls;
    assign {instruction_enable, read_enable, reg_write, reg_set, write_enable, jump_PC, branch_PC} = controls;
   
    // Logic
    always @(*)
        begin
            case(state)
                FETCH:
                    controls = 7'b1000000;
                DECODE:
                    begin
                        case(opcode)      
                            ADD:
                                controls = 7'b0000000;
                            SUB:
                                controls = 7'b0000000;
                            AND: 
                                controls = 7'b0000000;
                            IOR:
                                controls = 7'b0000000;
                            XOR:
                                controls = 7'b0000000;
                            SHR:
                                controls = 7'b0000000;
                            SHL:
                                controls = 7'b0000000;
                            CMP:
                                controls = 7'b0000000;
                            JMP:
                                controls = 7'b0000010;
                            BRZ:
                                controls = 7'b0000001;
                            BRN:
                                controls = 7'b0000001;
                            RES:
                                controls = 7'b0000000;
                            LOD:
                                controls = 7'b0000000;
                            STR:
                                controls = 7'b0000000;
                            SEL:
                                controls = 7'b0000000;
                            SEU:
                                controls = 7'b0000000;
                            default: // ???
                                controls = 7'b0000000;
                        endcase
                    end
                EXECUTE:
                    begin
                        case(opcode)      
                            ADD:
                                controls = 7'b000000;
                            SUB:
                                controls = 7'b0000000;
                            AND: 
                                controls = 7'b0000000;
                            IOR:
                                controls = 7'b0000000;
                            XOR:
                                controls = 7'b0000000;
                            SHR:
                                controls = 7'b0000000;
                            SHL:
                                controls = 7'b0000000;
                            CMP:
                                controls = 7'b0000000;
                            JMP:
                                controls = 7'b0000010;
                            BRZ:
                                controls = 7'b0000001;
                            BRN:
                                controls = 7'b0000001;
                            RES:
                                controls = 7'b0000000;
                            LOD:
                                controls = 7'b0100000;
                            STR:
                                controls = 7'b0000100;
                            SEL:
                                controls = 7'b0001000;
                            SEU:
                                controls = 7'b0001000;
                            default: // ???
                                controls = 7'b0000000;
                        endcase
                    end
                STORE:
                    begin
                        case(opcode)      
                            ADD:
                                controls = 7'b0010000;
                            SUB:
                                controls = 7'b0010000;
                            AND: 
                                controls = 7'b0010000;
                            IOR:
                                controls = 7'b0010000;
                            XOR:
                                controls = 7'b0010000;
                            SHR:
                                controls = 7'b0010000;
                            SHL:
                                controls = 7'b0010000;
                            CMP:
                                controls = 7'b0010000;
                            JMP:
                                controls = 7'b0010010;
                            BRZ:
                                controls = 7'b0000001;
                            BRN:
                                controls = 7'b0000001;
                            RES:
                                controls = 7'b0000000;
                            LOD:
                                controls = 7'b0110000;
                            STR:
                                controls = 7'b0000100;
                            SEL:
                                controls = 7'b0011000;
                            SEU:
                                controls = 7'b0011000;
                            default: // ???
                                controls = 7'b0000000;
                        endcase
                    end           
            endcase
        end

endmodule