; We first include useful definitions for our hardware (names of pins, ports, etc.)
; - The most common such definitions are listed in the m328pdef.inc file
.equ	PORTB, 0x05
.equ	DDRB,  0x04
.equ	UDRE0, 5		; USART Data Register Empty
.equ	UDR0, 0xc6	; MEMORY MAPPED

; ***** USART0 ***********************
; UDR0 - USART I/O Data Register
.equ	UDR0_0, 0	; USART I/O Data Register bit 0
.equ	UDR0_1, 1	; USART I/O Data Register bit 1
.equ	UDR0_2, 2	; USART I/O Data Register bit 2
.equ	UDR0_3, 3	; USART I/O Data Register bit 3
.equ	UDR0_4, 4	; USART I/O Data Register bit 4
.equ	UDR0_5, 5	; USART I/O Data Register bit 5
.equ	UDR0_6, 6	; USART I/O Data Register bit 6
.equ	UDR0_7, 7	; USART I/O Data Register bit 7

.equ	TXEN0, 3	; Transmitter Enable
.equ	RXEN0, 4	; Receiver Enable
.equ	F_CPU, 16000000		; CPU Frequency, 16 MHz

.equ	UBRR0L, 0xc4	; MEMORY MAPPED
.equ	UBRR0H, 0xc5	; MEMORY MAPPED
.equ	UCSR0C, 0xc2	; MEMORY MAPPED
.equ	UCSR0B, 0xc1	; MEMORY MAPPED
.equ	UCSR0A, 0xc0	; MEMORY MAPPED

.equ	TXC0, 6	; USART Transmitt Complete
.equ	RXC0, 7	; USART Receive Complete


; When the Reset button is pressed, the program counter is set to 0x0000, put an instruction there to jump to our code.
.ORG	0x0000					; The next instruction is written at 0x0000
RJMP	main					; Write the instruction to jump to "main" label

main:							; Sets all pins to "Output" mode
	LDI		r16, 0xFF			; Load the immedate value 0xFF (all bits 1) into register 16
	OUT		DDRB, r16			; Set Data Direction Register B (0x04) to output for all pins
	.equ	baud, 9600				; baudrate
	.equ	bps, (F_CPU/16/baud) - 1	; baud prescale
	ldi		r16, lo8(bps)			; load baud prescale
	ldi		r17, hi8(bps)			; into r17:r16
	rcall	initUART			; call initUART subroutine

loop:
	SBI		PORTB, 5			; Set the 5th bit in PortB (0x05). (i.e. turn on the LED)
	RCALL	delay				; Call the "delay" sub routine, stores current program location (+1) on the Stack
	CBI		PORTB, 5			; Clear the 5th bit in PortB (0x05). (i.e. turn off the LED)
	RCALL	delay				; Call the "delay" sub routine, stores current program location (+1) on the Stack
	rcall	getc				; retrieve character
	rcall	putc				; transmit character
	RJMP	loop				; Jump to "Loop" label (again)

; The Arduino clock is running at approximately 16 MHz (1 cycle = 62.5 nanoseconds)
; We can count from 0 until overflow using a pair of 8-bit registers (r24,r25) - 0 to 65536
; - This will take 65536 * (62.5 ns * # cycles per loop) = 16.384 ms
; 		(# cyles per loop = ADIW (2 cycles) + BRNE (2 cycles)  = 4 cycles)
; - If we repeat this counting, say 61 times, we get approx. 1 second of delay

delay:							; Delay sub-routine: delay for some number of clock cycles
	LDI	r16, 01					; Load r16 with 8 - number of times to repeat the outer loop

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

;**************************************************************
;* subroutine: initUART
;*
;* inputs: r17:r16 - baud rate prescale
;*
;* enables UART transmission with 8 data, 1 parity, no stop bit
;* at input baudrate
;*
;* registers modified: r16
;**************************************************************

initUART:
	sts	UBRR0L,r16			; load baud prescale
	sts	UBRR0H,r17			; to UBRR0

	ldi	r16,(1<<RXEN0)|(1<<TXEN0)	; enable transmitter
	sts	UCSR0B,r16			; and receiver

	ret					; return from subroutine

;**************************************************************
;* subroutine: putc
;*
;* inputs: r16 - character to transmit
;*
;* transmits single ASCII character via UART
;*
;* registers modified: r17
;**************************************************************

putc:	
	lds		r17, UCSR0A			; load UCSR0A into r17
	sbrs	r17, UDRE0			; wait for empty transmit buffer
	rjmp	putc				; repeat loop

	sts		UDR0, r16			; transmit character

	ret					; return from subroutine

;**************************************************************
;* subroutine: puts
;*
;* inputs: ZH:ZL - Program Memory address of string to transmit
;*
;* transmits null terminated string via UART
;*
;* registers modified: r16,r17,r30,r31
;**************************************************************

puts:
	lpm		r16, Z+				; load character from pmem
	cpi		r16, 0x00			; check if null
	breq	puts_end			; branch if null

puts_wait:
	lds		r17,UCSR0A			; load UCSR0A into r17
	sbrs	r17,UDRE0			; wait for empty transmit buffer
	rjmp	puts_wait			; repeat loop

	sts		UDR0,r16			; transmit character
	rjmp	puts				; repeat loop

puts_end:
	ret					; return from subroutine

;**************************************************************
;* subroutine: getc
;*
;* inputs: none
;*
;* outputs:	r16 - character received
;*
;* receives single ASCII character via UART
;*
;* registers modified: r16, r17
;**************************************************************

getc:	
	lds		r17,UCSR0A			; load UCSR0A into r17
	sbrs	r17,RXC0			; wait for empty receive buffer
	rjmp	getc				; repeat loop

	lds		r16, UDR0 			; get received character

	ret					; return from subroutine

;**************************************************************
;* subroutine: gets
;*
;* inputs: XH:XL - SRAM buffer address for rcv'd string
;*
;* outputs: none
;*
;* receives characters via UART and stores in data memory
;* until carriage return received
;*
;* registers modified: r16, r17, XL, XH
;**************************************************************

gets:	
	lds		r17,UCSR0A			; load UCSR0A into r17
	sbrs	r17,UDRE0			; wait for empty transmit buffer
	rjmp	putc				; repeat loop

	lds		r16, UDR0			; get received character

	cpi		r16,0x0D				; check if rcv'd char is CR
	breq	gets_end			; branch if CR rcv'd

	st		X+,r16				; store character to buffer
	rjmp	gets				; get another character

gets_end:
	ret					; return from subroutine
; FIN
