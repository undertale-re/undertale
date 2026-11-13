#include <stdio.h>
#include <string.h>

#define MAX_NAME   32
#define NUM_GRADES 3

typedef struct {
    int  id;
    char name[MAX_NAME];
    int  grades[NUM_GRADES];
} Student;

double student_average(const Student *s) {
    int sum = 0;
    for (int i = 0; i < NUM_GRADES; i++) {
        sum += s->grades[i];
    }
    return (double)sum / NUM_GRADES;
}

void student_print(const Student *s) {
    printf("ID %-3d %-10s grades: ", s->id, s->name);
    for (int i = 0; i < NUM_GRADES; i++) {
        printf("%3d ", s->grades[i]);
    }
    printf(" avg: %.1f\n", student_average(s));
}

/* Returns a pointer to the student with the highest average */
const Student *find_top_student(const Student list[], int count) {
    const Student *best = &list[0];
    for (int i = 1; i < count; i++) {
        if (student_average(&list[i]) > student_average(best)) {
            best = &list[i];
        }
    }
    return best;
}

int main(void) {
    Student class_list[] = {
        { 1, "Alice", { 88, 92, 79 } },
        { 2, "Bob",   { 72, 65, 80 } },
        { 3, "Chen",  { 95, 90, 98 } },
        { 4, "Dana",  { 60, 85, 77 } },
    };
    int count = sizeof(class_list) / sizeof(class_list[0]);

    for (int i = 0; i < count; i++) {
        student_print(&class_list[i]);
    }

    const Student *top = find_top_student(class_list, count);
    printf("\nTop student: %s (%.1f)\n", top->name, student_average(top));

    return 0;
}
