// Generate Immediate
// - Generates 32-bit immediate from the instruction for use by other modules (e.g. ALU) by...
//   - Extending the sign bit
//   - Padding with zeros (upper immediates)
module generate_immediate(instruction, immediate);
    
    // Declarations
    input [31:0] instruction;
    output reg [31:0] immediate;

    // Logic
    always @*
        begin
            case(instruction[6:0])
                7'b0000011,                                                                                                     // I-type (Loads): lw
                7'b0010011: immediate = {{20{instruction[31]}}, instruction[31:20]};                                            // I−type (ALU): add, sub, and, or
                7'b0100011: immediate = {{20{instruction[31]}}, instruction[31:25], instruction[11:7]};                         // S−type (stores): sw
                7'b1100011: immediate = {{20{instruction[31]}}, instruction[7], instruction[30:25],  instruction[11:8], 1'b0};  // B−type: beq
                7'b1101111: immediate = {{12{instruction[31]}}, instruction[19:12], instruction[20], instruction[30:21], 1'b0}; // J−type: jal
                7'b0110111: immediate = {instruction[31:12], 12'b0};                                                            // U−type: lui
                default: immediate = 32'bx; // Undefined
            endcase
        end

endmodule
