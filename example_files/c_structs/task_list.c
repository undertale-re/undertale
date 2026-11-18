#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { PRIORITY_LOW, PRIORITY_MEDIUM, PRIORITY_HIGH } Priority;

typedef struct {
    int      id;
    char     description[64];
    Priority priority;
    int      done;
} Task;

/* Self-referential struct: needs a tag name so it can refer to itself */
typedef struct TaskNode {
    Task             data;      /* struct embedded by value */
    struct TaskNode *next;      /* pointer to another node */
} TaskNode;

typedef struct {
    TaskNode *head;
    TaskNode *tail;
    int       count;
    int       next_id;
} TaskList;

void list_init(TaskList *list) {
    list->head = NULL;
    list->tail = NULL;
    list->count = 0;
    list->next_id = 1;
}

/* Append a task; returns its id, or -1 if allocation fails */
int list_add(TaskList *list, const char *description, Priority priority) {
    TaskNode *node = malloc(sizeof *node);
    if (node == NULL) {
        return -1;
    }
    node->data.id = list->next_id++;
    strncpy(node->data.description, description, sizeof node->data.description - 1);
    node->data.description[sizeof node->data.description - 1] = '\0';
    node->data.priority = priority;
    node->data.done = 0;
    node->next = NULL;

    if (list->tail == NULL) {
        list->head = node;
    } else {
        list->tail->next = node;
    }
    list->tail = node;
    list->count++;
    return node->data.id;
}

Task *list_find(TaskList *list, int id) {
    for (TaskNode *n = list->head; n != NULL; n = n->next) {
        if (n->data.id == id) {
            return &n->data;
        }
    }
    return NULL;
}

/* Remove a task by id; returns 0 on success */
int list_remove(TaskList *list, int id) {
    TaskNode *prev = NULL;
    TaskNode *cur = list->head;

    while (cur != NULL && cur->data.id != id) {
        prev = cur;
        cur = cur->next;
    }
    if (cur == NULL) {
        return -1;
    }

    if (prev == NULL) {
        list->head = cur->next;
    } else {
        prev->next = cur->next;
    }
    if (cur == list->tail) {
        list->tail = prev;
    }
    free(cur);
    list->count--;
    return 0;
}

void list_free(TaskList *list) {
    TaskNode *cur = list->head;
    while (cur != NULL) {
        TaskNode *next = cur->next;
        free(cur);
        cur = next;
    }
    list_init(list);
}

static const char *priority_name(Priority p) {
    switch (p) {
        case PRIORITY_HIGH:   return "HIGH";
        case PRIORITY_MEDIUM: return "MED";
        default:              return "LOW";
    }
}

void list_print(const TaskList *list) {
    printf("Tasks (%d):\n", list->count);
    for (const TaskNode *n = list->head; n != NULL; n = n->next) {
        printf("  [%c] #%d %-4s %s\n",
               n->data.done ? 'x' : ' ',
               n->data.id,
               priority_name(n->data.priority),
               n->data.description);
    }
}

int main(void) {
    TaskList list;
    list_init(&list);

    list_add(&list, "Write struct examples", PRIORITY_HIGH);
    int strip_id = list_add(&list, "Strip the binaries", PRIORITY_MEDIUM);
    list_add(&list, "Open in Binary Ninja", PRIORITY_MEDIUM);
    list_add(&list, "Water the plants", PRIORITY_LOW);

    list_find(&list, 1)->done = 1;
    list_print(&list);

    list_remove(&list, strip_id);
    printf("\nAfter removing #%d:\n", strip_id);
    list_print(&list);

    list_free(&list);
    return 0;
}
