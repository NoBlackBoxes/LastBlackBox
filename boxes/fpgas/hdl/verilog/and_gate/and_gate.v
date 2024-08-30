// AND Gate
module and_gate(a, b, y, gpio);

    // Declarations
    input a;
    input b;
    output y;
    output [12:0] gpio;

    // Blank GPIO
    assign gpio = 0;

    // Logic (Behaviour)
    assign y = a & b;   // a AND b

    // Logic (Gate) 
    // and(y, a, b);   // a AND b

endmodule