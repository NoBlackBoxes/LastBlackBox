/*
 * memset - fill memory with a constant byte
 *
 * Copyright (c) 2012-2022, Arm Limited.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

/* Assumptions:
 *
 * ARMv8-a, AArch64, Advanced SIMD, unaligned accesses.
 *
 */

#include "asmdefs.h"

#define dstin	x0  // Destination pointer
#define val	    w1  // 32-bit value to set
#define count	x2  // Number of bytes to set
#define dst	    x3  // Temporary register for destination manipulation
#define dstend	x4  // End address of the destination buffer

ENTRY (__memset_aarch64)
	PTR_ARG (0)
	SIZE_ARG (2)

	dup	v0.4S, val              // Duplicate the 32-bit value across the 128-bit vector register
	add	dstend, dstin, count    // Calculate the end address of the destination buffer
    mov dst, dstin              // Set dst to the start of the destination buffer
	str	q0, [dstin]             // Store initial 128-bit value

	sub	count, dstend, dst      // Adjust count for aligned destination
	sub	count, count, 16        // Adjust count for initial store

L(loop):
	stp	q0, q0, [dst, 32]! // Store 256 bits and update destination pointer
	subs	count, count, 32 // Subtract 32 from count and update flags
	b.hi	L(loop) // Repeat loop if count > 0

	ret

END (__memset_aarch64)
