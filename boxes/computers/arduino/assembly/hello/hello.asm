; Your first assembly program (for an AVR microcontroller)
; This code is for the Atmel Mega 328P (used by the Arduino Nano)

; We first include useful definitions for our hardware (names of pins, ports, etc.)
; - The most common such definitions are listed in the m328pdef.inc file
.equ	PORTB,0x05
.equ	DDRB, 0x04

; When the Reset button is pressed, the program counter is set to 0x0000, put an instruction there to jump to our code.
.ORG	0x0000					; The next instruction is written at 0x0000
RJMP	main					; Write the instruction to jump to "main" label

main:							; Sets all pins to "Output" mode
	LDI		r16, 0xFF			; Load the immedate value 0xFF (all bits 1) into register 16
	OUT		DDRB, r16			; Set Data Direction Register B (0x04) to output for all pins

loop:
	SBI		PORTB, 5			; Set the 5th bit in PortB (0x05). (i.e. turn on the LED)
	RCALL	delay				; Call the "delay" sub routine, stores current program location (+1) on the Stack
	CBI		PORTB, 5			; Clear the 5th bit in PortB (0x05). (i.e. turn off the LED)
	RCALL	delay				; Call the "delay" sub routine, stores current program location (+1) on the Stack
	RJMP	loop				; Jump to "Loop" label (again)


; The Arduino clock is running at approximately 16 MHz (1 cycle = 62.5 nanoseconds)
; We can count from 0 until overflow using a pair of 8-bit registers (r24,r25) - 0 to 65536
; - This will take 65536 * (62.5 ns * # cycles per loop) = 16.384 ms
; 		(# cyles per loop = ADIW (2 cycles) + BRNE (2 cycles)  = 4 cycles)
; - If we repeat this counting, say 61 times, we get approx. 1 second of delay

delay:							; Delay sub-routine: delay for some number of clock cycles
	LDI	r16, 61					; Load r16 with 8 - number of times to repeat the outer loop

	outer_loop:
	LDI	r24, lo8(0)				; Load an offset number in r24 (lower byte)
	LDI	r25, hi8(0)			    ; Load an offset number in r25 (upper byte)

	delay_loop:					; Enter inner delay loop and burn clock cyles
	ADIW	r24, 1				; add 1 to r24, set "zero flag" in status register if the result is zero
	BRNE	delay_loop			; repeat (i.e. branch to start of inner loop) if not equal (test zero flag, will skip if r24/25 is zero)
	DEC		r16					; decrement outer loop counter, set "zero flag" in status register if the result is zero
	BRNE	outer_loop			; repeat (i.e. branch to start of outer loop) if not equal (test zero flag, will skip if r16 is zero)
	
	RET							; Return from "delay" sub-routine. The program counter is restored from the Stack.
; FIN
