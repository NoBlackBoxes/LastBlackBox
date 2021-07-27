// AND Gate
module and_gate(a, b, y);

    // Declarations
    input a;
    input b;
    output y;

    // Logic (Behaviour)
    assign y = a & b;   // a AND b

    // Logic (Gate) 
    // and(y, a, b);   // a AND b

endmodule