// Extend
// - Extends the sign bit of subsets of immediates in the instruction to 32-bits, for use by 32-bit modules (e.g. ALU)
module extend(instruction, immediate_select, immediate_extended);
    
    // Declarations
    input [31:0] instruction;
    input [1:0] immediate_select;
    output reg [31:0] immediate_extended;

    // Logic
    always @*
        begin
            case(immediate_select)
                2'b00: immediate_extended = {{20{instruction[31]}}, instruction[31:20]}; // I−type
                2'b01: immediate_extended = {{20{instruction[31]}}, instruction[31:25], instruction[11:7]}; // S−type (stores)
                2'b10: immediate_extended = {{20{instruction[31]}}, instruction[7], instruction[30:25],  instruction[11:8], 1'b0}; // B−type (branches)
                2'b11: immediate_extended = {{12{instruction[31]}}, instruction[19:12], instruction[20], instruction[30:21], 1'b0}; // J−type (jal)
                default: immediate_extended = 32'bx; // Undefined
            endcase
        end

endmodule