#include <stdio.h>
#include <string.h>

#define MAX_BOOKS 8
#define MAX_TEXT  48

typedef struct {
    int year;
    int month;
    int day;
} Date;

typedef struct {
    char first_name[MAX_TEXT];
    char last_name[MAX_TEXT];
    Date born;                  /* struct inside a struct */
} Author;

typedef struct {
    char   title[MAX_TEXT];
    Author author;              /* struct (which itself contains a struct) */
    Date   published;
    int    pages;
    int    checked_out;         /* 0 = available, 1 = checked out */
} Book;

typedef struct {
    char name[MAX_TEXT];
    Book books[MAX_BOOKS];      /* array of structs inside a struct */
    int  book_count;
} Library;

int library_add(Library *lib, Book book) {
    if (lib->book_count >= MAX_BOOKS) {
        return -1;
    }
    lib->books[lib->book_count++] = book;   /* whole struct copied */
    return 0;
}

Book *library_find(Library *lib, const char *title) {
    for (int i = 0; i < lib->book_count; i++) {
        if (strcmp(lib->books[i].title, title) == 0) {
            return &lib->books[i];
        }
    }
    return NULL;
}

int library_checkout(Library *lib, const char *title) {
    Book *b = library_find(lib, title);
    if (b == NULL || b->checked_out) {
        return -1;
    }
    b->checked_out = 1;
    return 0;
}

/* Age of the author when the book came out (chains through two levels) */
int author_age_at_publication(const Book *b) {
    int age = b->published.year - b->author.born.year;
    if (b->published.month < b->author.born.month ||
        (b->published.month == b->author.born.month &&
         b->published.day < b->author.born.day)) {
        age--;
    }
    return age;
}

void book_print(const Book *b) {
    printf("  %-24s %s %s, %04d-%02d-%02d, %d pages%s\n",
           b->title,
           b->author.first_name, b->author.last_name,
           b->published.year, b->published.month, b->published.day,
           b->pages,
           b->checked_out ? " [checked out]" : "");
}

int main(void) {
    Library lib = { .name = "Riverside Branch", .book_count = 0 };

    /* Designated initializers make nested structs readable */
    library_add(&lib, (Book){
        .title     = "Frankenstein",
        .author    = { "Mary", "Shelley", { 1797, 8, 30 } },
        .published = { 1818, 1, 1 },
        .pages     = 280,
    });
    library_add(&lib, (Book){
        .title     = "Pride and Prejudice",
        .author    = { "Jane", "Austen", { 1775, 12, 16 } },
        .published = { 1813, 1, 28 },
        .pages     = 432,
    });
    library_add(&lib, (Book){
        .title     = "Great Expectations",
        .author    = { "Charles", "Dickens", { 1812, 2, 7 } },
        .published = { 1861, 8, 1 },
        .pages     = 544,
    });

    library_checkout(&lib, "Frankenstein");

    printf("%s (%d books)\n", lib.name, lib.book_count);
    for (int i = 0; i < lib.book_count; i++) {
        book_print(&lib.books[i]);
    }

    printf("\nAuthor ages at publication:\n");
    for (int i = 0; i < lib.book_count; i++) {
        printf("  %-10s %d\n", lib.books[i].author.last_name,
               author_age_at_publication(&lib.books[i]));
    }

    printf("\nsizeof(Date)=%zu  sizeof(Author)=%zu  sizeof(Book)=%zu  sizeof(Library)=%zu\n",
           sizeof(Date), sizeof(Author), sizeof(Book), sizeof(Library));
    return 0;
}
