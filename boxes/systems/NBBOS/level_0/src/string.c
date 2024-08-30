#include "common.h"
#include "string.h"

// Function Definitions

/* strlen: count the characters in s, not including NULL */
uint32_t strlen(const char *s)
{
    uint32_t i;
    for (i = 0; s[i] != '\0'; i++)
    {
    }
    return i;
}

/* reverse: reverse string s in place */
void reverse(char *s)
{
    uint32_t i, j;
    char c;

    for (i = 0, j = strlen(s) - 1; i < j; i++, j--)
    {
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}

/* itoa:  convert n to characters in s */
void itoa(int64_t n, char *s)
{
    int64_t i, sign;

    sign = n;
    if (n < 0)          /* record sign */
        n = -n;         /* make n positive */
    i = 0;
    do
    {                          /* generate digits in reverse order */
        s[i++] = n % 10 + '0'; /* get next digit */
    } while ((n /= 10) > 0);   /* delete it */
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    reverse(s);
}

/* format: format report of named value */
void format(const char *name, int64_t value, char *s)
{
    uint32_t n = strlen(name);
    uint32_t i = 0;
    for (i = 0; i < n; i++)
    {
        s[i] = name[i];     /// This breaks at O3 and O4 optimizationlevels!
    }
    s[i++] = ':';
    s[i++] = ' ';
    itoa(value, &(s[i]));
    
    return;
}
