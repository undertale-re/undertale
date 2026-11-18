#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[32];
    char country[24];
    int  lead_time_days;
} Supplier;

typedef struct {
    char     sku[12];
    char     name[40];
    int      quantity;
    double   unit_price;
    Supplier supplier;          /* nested struct */
} Item;

typedef struct {
    Item  *items;               /* heap array of structs */
    size_t count;
    size_t capacity;
} Inventory;

int inventory_init(Inventory *inv, size_t initial_capacity) {
    inv->items = malloc(initial_capacity * sizeof *inv->items);
    if (inv->items == NULL) {
        return -1;
    }
    inv->count = 0;
    inv->capacity = initial_capacity;
    return 0;
}

/* Double the capacity when full */
static int inventory_grow(Inventory *inv) {
    size_t new_capacity = inv->capacity * 2;
    Item *bigger = realloc(inv->items, new_capacity * sizeof *bigger);
    if (bigger == NULL) {
        return -1;
    }
    inv->items = bigger;
    inv->capacity = new_capacity;
    printf("  (grew capacity to %zu)\n", new_capacity);
    return 0;
}

int inventory_add(Inventory *inv, const Item *item) {
    if (inv->count == inv->capacity && inventory_grow(inv) != 0) {
        return -1;
    }
    inv->items[inv->count++] = *item;
    return 0;
}

double inventory_total_value(const Inventory *inv) {
    double total = 0.0;
    for (size_t i = 0; i < inv->count; i++) {
        total += inv->items[i].quantity * inv->items[i].unit_price;
    }
    return total;
}

/* Items at or below a threshold, with their supplier's lead time */
void inventory_report_low_stock(const Inventory *inv, int threshold) {
    printf("Low stock (<= %d):\n", threshold);
    for (size_t i = 0; i < inv->count; i++) {
        const Item *it = &inv->items[i];
        if (it->quantity <= threshold) {
            printf("  %s %-18s qty %3d  reorder from %s (%s), %d days\n",
                   it->sku, it->name, it->quantity,
                   it->supplier.name, it->supplier.country,
                   it->supplier.lead_time_days);
        }
    }
}

void inventory_free(Inventory *inv) {
    free(inv->items);
    inv->items = NULL;
    inv->count = inv->capacity = 0;
}

int main(void) {
    Supplier acme   = { "Acme Fasteners", "USA",     5 };
    Supplier nordic = { "Nordic Metals",  "Sweden", 21 };
    Supplier pacif  = { "Pacific Supply", "Japan",  14 };

    Item stock[] = {
        { "BLT-0001", "Hex bolt M8",      450, 0.12, acme   },
        { "NUT-0002", "Lock nut M8",        8, 0.09, acme   },
        { "SHT-0003", "Steel sheet 2mm",    3, 42.5, nordic },
        { "BRG-0004", "Ball bearing 608",  60, 1.35, pacif  },
        { "WSH-0005", "Washer M8",        900, 0.03, acme   },
        { "ROD-0006", "Steel rod 10mm",     5, 7.80, nordic },
    };
    size_t stock_count = sizeof stock / sizeof stock[0];

    Inventory inv;
    if (inventory_init(&inv, 2) != 0) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }

    printf("Adding %zu items:\n", stock_count);
    for (size_t i = 0; i < stock_count; i++) {
        if (inventory_add(&inv, &stock[i]) != 0) {
            fprintf(stderr, "Out of memory\n");
            inventory_free(&inv);
            return 1;
        }
    }

    printf("\n%zu items, capacity %zu, total value $%.2f\n\n",
           inv.count, inv.capacity, inventory_total_value(&inv));
    inventory_report_low_stock(&inv, 10);

    inventory_free(&inv);
    return 0;
}
