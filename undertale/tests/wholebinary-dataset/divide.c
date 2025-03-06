#include <stdio.h>

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
    printf("division: %.2f\\n", divide((float)num2, (float)num1));
    return 0;
}