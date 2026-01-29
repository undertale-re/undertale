int main() {
    #if (defined(__i386__) || defined(__x86_64__))
        __asm__(".byte 0xFE");
    #else
        #error "unsupported architecture (requires x86)"
    #endif

    return 0;
}
