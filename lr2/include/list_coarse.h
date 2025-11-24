#ifndef LIST_COARSE_H
#define LIST_COARSE_H

#include <mutex>

class CoarseList {
public:
    CoarseList();
    ~CoarseList();

    bool insert(int value);
    bool remove(int value);
    bool find(int value);

private:
    struct Node {
        int value;
        Node* next;
        Node(int v): value(v), next(nullptr) {}
    };
    Node* head;
    std::mutex mtx;
};

#endif // LIST_COARSE_H
