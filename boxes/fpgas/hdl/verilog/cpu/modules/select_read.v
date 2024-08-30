// Select Read
// - Selects appropriate sized (and extended) 32-bit data read from memory for register load instructions
module select_read(funct3, read_data_in, read_data_out);
    
    // Declarations
    input [2:0] funct3;
    input [31:0] read_data_in;
    output reg [31:0] read_data_out;

    // Logic
    always @*
        begin
            case(funct3)
                3'b000: read_data_out = {{24{read_data_in[7]}}, read_data_in[7:0]};     // lb
                3'b001: read_data_out = {{16{read_data_in[15]}}, read_data_in[15:0]};   // lh
                3'b010: read_data_out = read_data_in;                                   // lw
                3'b100: read_data_out = {24'b0, read_data_in[7:0]};                     // lbu
                3'b101: read_data_out = {16'b0, read_data_in[15:0]};                    // lhu
                default: read_data_out = 32'bx; // Undefined
            endcase
        end

endmodule
