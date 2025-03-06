#include <stdio.h>

int square(int x) {
    return x * x;
}

int main() {
    int num = 2;
    printf("Square: %d\\n", square(num));
    return 0;
}