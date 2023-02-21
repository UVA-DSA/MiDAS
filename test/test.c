#include <stdio.h>
int main() {
    size_t mask = (1 << 6) - 1;
    printf("%zx", (0xffffffffffffffff & mask));
    return 1;
}
