SFILES = $(wildcard src/*.S)
CFILES = $(wildcard src/*.c)
OFILES = $(subst src,build/src,$(CFILES:.c=.o))
AFILES = $(subst src,build/src,$(CFILES:.c=.S))
OFILES += $(subst src,build/src,$(SFILES:.S=.o))
$(info CFILES is $(CFILES))
$(info OFILES is $(OFILES))
GCCFLAGS = -mcpu=cortex-a72 -mtune=cortex-a72 -O2 -Wall -Wextra -ffreestanding -nostdlib -nostartfiles -Iinclude # -fsanitize-undefined-trap-on-error
ASMFLAGS = -mcpu=cortex-a72 -mtune=cortex-a72 -O4 -Iinclude
GCCPATH = ../_tmp/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-elf/bin

all: clean kernel8.img

build/%.o: %.S
	$(GCCPATH)/aarch64-none-elf-gcc $(GCCFLAGS) -c $< -o $@

build/%.o: %.c
	$(GCCPATH)/aarch64-none-elf-gcc $(GCCFLAGS) -c $< -o $@

build/%.S: %.c
	$(GCCPATH)/aarch64-none-elf-gcc $(GCCFLAGS) -S -c $< -o $@

kernel8.img: build/src/boot.o $(OFILES) $(AFILES)
	$(GCCPATH)/aarch64-none-elf-ld -nostdlib $(OFILES) -T link.ld -o build/kernel8.elf
	$(GCCPATH)/aarch64-none-elf-objcopy -O binary build/kernel8.elf build/kernel8.img

clean:
	/bin/rm -rf build/*
	/bin/mkdir build/src
