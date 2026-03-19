#include <stdint.h>

typedef struct {
    uint8_t pad;
    uint32_t value;
} Datum;

volatile Datum data[8];

int main() {
    if (data[2].value > 2) {
        return 1;
    }

    return 0;
}
