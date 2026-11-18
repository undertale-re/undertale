#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char   owner[48];
    long   account_number;
    double balance;
} BankAccount;

/* Allocate and initialise a new account; returns NULL on failure */
BankAccount *account_create(const char *owner, long number, double initial) {
    BankAccount *acc = malloc(sizeof *acc);
    if (acc == NULL) {
        return NULL;
    }
    strncpy(acc->owner, owner, sizeof acc->owner - 1);
    acc->owner[sizeof acc->owner - 1] = '\0';
    acc->account_number = number;
    acc->balance = initial;
    return acc;
}

void account_destroy(BankAccount *acc) {
    free(acc);
}

int account_deposit(BankAccount *acc, double amount) {
    if (amount <= 0) {
        return -1;
    }
    acc->balance += amount;
    return 0;
}

int account_withdraw(BankAccount *acc, double amount) {
    if (amount <= 0 || amount > acc->balance) {
        return -1;
    }
    acc->balance -= amount;
    return 0;
}

void account_print(const BankAccount *acc) {
    printf("[%ld] %s: $%.2f\n", acc->account_number, acc->owner, acc->balance);
}

int main(void) {
    BankAccount *acc = account_create("Jordan Smith", 100245, 50.00);
    if (acc == NULL) {
        fprintf(stderr, "Failed to allocate account\n");
        return 1;
    }

    account_print(acc);

    account_deposit(acc, 125.50);
    printf("After depositing $125.50:  ");
    account_print(acc);

    if (account_withdraw(acc, 500.00) != 0) {
        printf("Withdrawal of $500.00 refused: insufficient funds\n");
    }

    account_withdraw(acc, 75.25);
    printf("After withdrawing $75.25:  ");
    account_print(acc);

    account_destroy(acc);
    return 0;
}
