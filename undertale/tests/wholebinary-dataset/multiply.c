#include <stdio.h>

int multiply(int a, int b) {
    return a * b;
}

float divide(float a, float b) {
    if (b == 0) {
        printf("Division by zero error!\\n");
        return 0;
    }
    return a / b;
}

int main() {
    int num1 = 5;
    int num2 = 10;
    printf("multiplication: %d\\n", multiply(num1, num2));
    return 0;
}