int main() {
    #if (defined(__i386__) || defined(__x86_64__))
        __asm__("call .+5");
    #else
        #error "unsupported architecture (requires x86)"
    #endif

    return 0;
}
