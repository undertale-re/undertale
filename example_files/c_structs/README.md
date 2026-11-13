# Example Files
Files listed in order of increasing struct complexity
## point.c
A simple 2D point struct with a few geometry helpers.  
Struct contains two doubles

## bank_account.c
A bank account struct allocated on the heap, with deposit/withdraw operations and error checks.  
Struct contains a string, long, and double.

## students.c
Store student records in an array of structs, compute averages, and find the top student.  
Struct contains two ints and a string.

## shapes.c
A struct containing a union of other structs (a "tagged union"), and a struct of function pointers used like a simple vtable. Also shows a Drawing struct holding an array of pointers to shapes.  
```
Shape { ShapeKind kind; const ShapeOps *ops; Point origin;
        union { Circle circle; Rect rect; Triangle tri; } as; }
ShapeOps { area(); perimeter(); name }
```

## task_list.c
A linked list of tasks. Each node is a struct that points to another struct of the same type, and a separate container struct tracks the list as a whole.  
```
TaskList { head, tail, count }  
   head -> TaskNode -> TaskNode -> TaskNode -> NULL  
           { Task data; TaskNode *next; }
```

## inventory.c
A struct that owns a dynamically growing array of structs (like a vector/ArrayList), where each element also contains a nested struct.  
```
 Inventory { Item *items; size_t count; size_t capacity; }  
    items -> [ Item | Item | Item | ... ]   (heap, grown with realloc)  
               └── Supplier supplier
```

## library.c
Structs nested inside structs, plus a fixed-size array of structs embedded in another struct.  
```
 Library  
  └── Book books[MAX_BOOKS]  
      ├── Author author  
      │    └── Date born  
      └── Date published  
```

## contacts_tree.c  
A binary search tree of contacts, sorted by name. Each tree node points to two other nodes, and each contact holds a small array of PhoneNumber structs.  
```
ContactBook { ContactNode *root; int size; }  
  root -> ContactNode { Contact contact; left; right; }  
             └── Contact { name; PhoneNumber phones[3];   phone_count; }  
```