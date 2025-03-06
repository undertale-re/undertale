#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int num1 = 5;
    int num2 = 10;
    printf("sum: %d\\n", add(num1, num2));
    return 0;
}