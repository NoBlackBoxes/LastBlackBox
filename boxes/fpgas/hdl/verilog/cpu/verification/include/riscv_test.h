#ifndef _CPU_TEST_H
#define _CPU_TEST_H

#define RVTEST_RV32U
#define TESTNUM x28
#define DEBUG_REG 0xFFFFFFF0

#define RVTEST_CODE_BEGIN		\
	.text;				\
	.global test;			\
test:

#define RVTEST_PASS			\
.pass:					\
	addi 	a0, x0, DEBUG_REG;		\
	addi	a1, x0, 0x1;			\
	sw		a1, 0(a0);		\
	j .pass;

#define RVTEST_FAIL			\
.fail:					\
	addi 	a0, x0, DEBUG_REG;		\
	addi	a1, x0, 0x0;			\
	sw		a1, 0(a0);		\
	j .fail;

#define RVTEST_CODE_END
#define RVTEST_DATA_BEGIN .balign 4;
#define RVTEST_DATA_END

#endif
