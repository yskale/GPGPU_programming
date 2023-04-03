#include <stdarg.h>
#include <stdio.h>

long sum(long num, ...)
{
    va_list ap;
    long sum = 0;

    va_start(ap, num);

    while (num--)
    {
        sum += va_arg(ap, long);
    }

    va_end(ap);
    return sum;
}

int main()
{
    long ret;
    ret = sum(8L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L);
    printf("sum(1..8) = %ld\n", ret);

    return 0;
}