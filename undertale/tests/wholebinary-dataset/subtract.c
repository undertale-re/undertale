#include <stdio.h>

int subtract(int a, int b) {
    return a - b;
}

int main() {
    int num1 = 5;
    int num2 = 10;
    printf("subtract: %d\\n", subtract(num1, num2));
    return 0;
}