#ifndef LIST_COARSE_H
#define LIST_COARSE_H

#include <mutex>

class CoarseList {
public:
    CoarseList();
    ~CoarseList();

    // Insert value if not present. Returns true if inserted.
    bool insert(int value);

    // Delete value if present. Returns true if deleted.
    bool remove(int value);

    // Find value.
    bool find(int value);

private:
    struct Node {
        int value;
        Node* next;
        Node(int v): value(v), next(nullptr) {}
    };
    Node* head;
    std::mutex mtx; // single coarse mutex
};

#endif // LIST_COARSE_H

