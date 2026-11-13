#include <stdio.h>
#include <math.h>

typedef struct {
    double x;
    double y;
} Point;

/* Euclidean distance between two points */
double point_distance(Point a, Point b) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    return sqrt(dx * dx + dy * dy);
}

/* Point halfway between a and b */
Point point_midpoint(Point a, Point b) {
    Point mid = { (a.x + b.x) / 2.0, (a.y + b.y) / 2.0 };
    return mid;
}

/* Move a point in place (passed by pointer) */
void point_translate(Point *p, double dx, double dy) {
    p->x += dx;
    p->y += dy;
}

void point_print(const char *label, Point p) {
    printf("%s = (%.2f, %.2f)\n", label, p.x, p.y);
}

int main(void) {
    Point a = { 0.0, 0.0 };
    Point b = { 3.0, 4.0 };

    point_print("a", a);
    point_print("b", b);
    printf("distance(a, b) = %.2f\n", point_distance(a, b));
    point_print("midpoint", point_midpoint(a, b));

    point_translate(&a, 1.5, -2.0);
    point_print("a after translate", a);

    return 0;
}
