long double x = 3.14L;

int main(void)
{
    #if (defined(__i386__) || defined(__x86_64__))
        __asm__ volatile (
            "fldt %0\n"
            :
            : "m"(x)
        );
    #else
        #error "unsupported architecture (requires x86)"
    #endif

    return 0;
}
