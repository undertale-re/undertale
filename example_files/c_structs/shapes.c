#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct { double x, y; } Point;

typedef struct { double radius; } Circle;
typedef struct { double width, height; } Rect;
typedef struct { Point a, b, c; } Triangle;    /* structs inside a struct */

typedef enum { SHAPE_CIRCLE, SHAPE_RECT, SHAPE_TRIANGLE } ShapeKind;

typedef struct Shape Shape;

/* A struct whose members are function pointers */
typedef struct {
    const char *name;
    double (*area)(const Shape *s);
    double (*perimeter)(const Shape *s);
} ShapeOps;

struct Shape {
    ShapeKind       kind;     /* tag: says which union member is valid */
    const ShapeOps *ops;
    Point           origin;
    union {                   /* only one of these is in use at a time */
        Circle   circle;
        Rect     rect;
        Triangle tri;
    } as;
};

/* ---- Circle ---- */
static double circle_area(const Shape *s) {
    return M_PI * s->as.circle.radius * s->as.circle.radius;
}
static double circle_perimeter(const Shape *s) {
    return 2.0 * M_PI * s->as.circle.radius;
}

/* ---- Rectangle ---- */
static double rect_area(const Shape *s) {
    return s->as.rect.width * s->as.rect.height;
}
static double rect_perimeter(const Shape *s) {
    return 2.0 * (s->as.rect.width + s->as.rect.height);
}

/* ---- Triangle ---- */
static double dist(Point p, Point q) {
    return hypot(q.x - p.x, q.y - p.y);
}
static double tri_area(const Shape *s) {
    const Triangle *t = &s->as.tri;
    return fabs((t->b.x - t->a.x) * (t->c.y - t->a.y) -
                (t->c.x - t->a.x) * (t->b.y - t->a.y)) / 2.0;
}
static double tri_perimeter(const Shape *s) {
    const Triangle *t = &s->as.tri;
    return dist(t->a, t->b) + dist(t->b, t->c) + dist(t->c, t->a);
}

/* One shared, read-only operations table per shape kind */
static const ShapeOps CIRCLE_OPS   = { "circle",    circle_area, circle_perimeter };
static const ShapeOps RECT_OPS     = { "rectangle", rect_area,   rect_perimeter   };
static const ShapeOps TRIANGLE_OPS = { "triangle",  tri_area,    tri_perimeter    };

Shape *shape_circle(Point origin, double radius) {
    Shape *s = malloc(sizeof *s);
    if (s) {
        s->kind = SHAPE_CIRCLE;
        s->ops = &CIRCLE_OPS;
        s->origin = origin;
        s->as.circle.radius = radius;
    }
    return s;
}

Shape *shape_rect(Point origin, double width, double height) {
    Shape *s = malloc(sizeof *s);
    if (s) {
        s->kind = SHAPE_RECT;
        s->ops = &RECT_OPS;
        s->origin = origin;
        s->as.rect = (Rect){ width, height };
    }
    return s;
}

Shape *shape_triangle(Point a, Point b, Point c) {
    Shape *s = malloc(sizeof *s);
    if (s) {
        s->kind = SHAPE_TRIANGLE;
        s->ops = &TRIANGLE_OPS;
        s->origin = a;
        s->as.tri = (Triangle){ a, b, c };
    }
    return s;
}

/* A struct holding an array of pointers to structs */
#define MAX_SHAPES 8
typedef struct {
    const char *title;
    Shape      *shapes[MAX_SHAPES];
    int         count;
} Drawing;

int drawing_add(Drawing *d, Shape *s) {
    if (s == NULL || d->count >= MAX_SHAPES) {
        free(s);
        return -1;
    }
    d->shapes[d->count++] = s;
    return 0;
}

void drawing_print(const Drawing *d) {
    double total_area = 0.0;
    printf("%s:\n", d->title);
    for (int i = 0; i < d->count; i++) {
        const Shape *s = d->shapes[i];
        double area = s->ops->area(s);           /* call through pointer */
        total_area += area;
        printf("  %-10s at (%5.1f, %5.1f)  area %8.2f  perimeter %7.2f\n",
               s->ops->name, s->origin.x, s->origin.y,
               area, s->ops->perimeter(s));
    }
    printf("  total area: %.2f\n", total_area);
}

void drawing_free(Drawing *d) {
    for (int i = 0; i < d->count; i++) {
        free(d->shapes[i]);
    }
    d->count = 0;
}

int main(void) {
    Drawing d = { .title = "Sketch", .count = 0 };

    drawing_add(&d, shape_circle((Point){ 0, 0 }, 2.5));
    drawing_add(&d, shape_rect((Point){ 4, 1 }, 3, 5));
    drawing_add(&d, shape_triangle((Point){ 0, 0 }, (Point){ 4, 0 }, (Point){ 0, 3 }));
    drawing_add(&d, shape_circle((Point){ -3, 2 }, 1));

    drawing_print(&d);

    printf("\nsizeof(Circle)=%zu  sizeof(Rect)=%zu  sizeof(Triangle)=%zu  sizeof(Shape)=%zu\n",
           sizeof(Circle), sizeof(Rect), sizeof(Triangle), sizeof(Shape));

    drawing_free(&d);
    return 0;
}
