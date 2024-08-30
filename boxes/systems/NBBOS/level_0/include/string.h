#pragma once
#include "common.h"

// Definitions
#define MAX_STRING 256

// Declartions
uint32_t strlen(const char *s);
void reverse(char *s);
void itoa(int64_t n, char *s);
void format(const char *name, int64_t code, char *s);
