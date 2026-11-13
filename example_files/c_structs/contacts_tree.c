#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PHONES 3

typedef enum { PHONE_MOBILE, PHONE_HOME, PHONE_WORK } PhoneType;

typedef struct {
    PhoneType type;
    char      number[20];
} PhoneNumber;

typedef struct {
    char        name[40];
    char        email[48];
    PhoneNumber phones[MAX_PHONES];   /* array of structs inside a struct */
    int         phone_count;
} Contact;

typedef struct ContactNode {
    Contact             contact;
    struct ContactNode *left;
    struct ContactNode *right;
} ContactNode;

typedef struct {
    ContactNode *root;
    int          size;
} ContactBook;

static ContactNode *node_new(const char *name, const char *email) {
    ContactNode *n = calloc(1, sizeof *n);   /* zeroes all fields */
    if (n == NULL) {
        return NULL;
    }
    strncpy(n->contact.name, name, sizeof n->contact.name - 1);
    strncpy(n->contact.email, email, sizeof n->contact.email - 1);
    return n;
}

/* Insert (or find existing) contact; returns pointer to it, or NULL */
Contact *book_insert(ContactBook *book, const char *name, const char *email) {
    ContactNode **link = &book->root;   /* pointer to the pointer we may set */

    while (*link != NULL) {
        int cmp = strcmp(name, (*link)->contact.name);
        if (cmp == 0) {
            return &(*link)->contact;
        }
        link = (cmp < 0) ? &(*link)->left : &(*link)->right;
    }

    *link = node_new(name, email);
    if (*link == NULL) {
        return NULL;
    }
    book->size++;
    return &(*link)->contact;
}

Contact *book_find(const ContactBook *book, const char *name) {
    ContactNode *n = book->root;
    while (n != NULL) {
        int cmp = strcmp(name, n->contact.name);
        if (cmp == 0) {
            return &n->contact;
        }
        n = (cmp < 0) ? n->left : n->right;
    }
    return NULL;
}

int contact_add_phone(Contact *c, PhoneType type, const char *number) {
    if (c->phone_count >= MAX_PHONES) {
        return -1;
    }
    PhoneNumber *p = &c->phones[c->phone_count++];
    p->type = type;
    strncpy(p->number, number, sizeof p->number - 1);
    p->number[sizeof p->number - 1] = '\0';
    return 0;
}

static const char *phone_type_name(PhoneType t) {
    switch (t) {
        case PHONE_MOBILE: return "mobile";
        case PHONE_HOME:   return "home";
        default:           return "work";
    }
}

static void contact_print(const Contact *c) {
    printf("  %-16s %s\n", c->name, c->email);
    for (int i = 0; i < c->phone_count; i++) {
        printf("      %-6s %s\n", phone_type_name(c->phones[i].type),
               c->phones[i].number);
    }
}

/* In-order traversal visits nodes alphabetically */
static void print_in_order(const ContactNode *n) {
    if (n == NULL) {
        return;
    }
    print_in_order(n->left);
    contact_print(&n->contact);
    print_in_order(n->right);
}

static int tree_height(const ContactNode *n) {
    if (n == NULL) {
        return 0;
    }
    int l = tree_height(n->left);
    int r = tree_height(n->right);
    return 1 + (l > r ? l : r);
}

static void free_nodes(ContactNode *n) {
    if (n == NULL) {
        return;
    }
    free_nodes(n->left);
    free_nodes(n->right);
    free(n);
}

void book_free(ContactBook *book) {
    free_nodes(book->root);
    book->root = NULL;
    book->size = 0;
}

int main(void) {
    ContactBook book = { NULL, 0 };

    Contact *c;
    c = book_insert(&book, "Morgan Lee", "morgan@example.com");
    contact_add_phone(c, PHONE_MOBILE, "555-0142");
    contact_add_phone(c, PHONE_WORK,   "555-0199");

    c = book_insert(&book, "Ana Ruiz", "ana@example.com");
    contact_add_phone(c, PHONE_HOME, "555-0107");

    c = book_insert(&book, "Sam Okafor", "sam@example.com");
    contact_add_phone(c, PHONE_MOBILE, "555-0175");

    c = book_insert(&book, "Divya Rao", "divya@example.com");
    contact_add_phone(c, PHONE_MOBILE, "555-0121");
    contact_add_phone(c, PHONE_HOME,   "555-0133");
    contact_add_phone(c, PHONE_WORK,   "555-0188");

    book_insert(&book, "Ana Ruiz", "duplicate@example.com");  /* ignored */

    printf("Contacts (%d, tree height %d):\n", book.size, tree_height(book.root));
    print_in_order(book.root);

    const char *lookup = "Sam Okafor";
    Contact *found = book_find(&book, lookup);
    printf("\nLookup \"%s\": %s\n", lookup, found ? found->email : "not found");

    book_free(&book);
    return 0;
}
