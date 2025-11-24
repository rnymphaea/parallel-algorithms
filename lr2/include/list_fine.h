#ifndef LIST_FINE_H
#define LIST_FINE_H

#include <mutex>
#include <atomic>

class FineList {
public:
    FineList();
    ~FineList();

    bool insert(int value);
    bool remove(int value);
    bool find(int value) const;

private:
    struct Node {
        int value;
        Node* next;
        std::mutex mtx;
        std::atomic<bool> marked;

        Node(int v) : value(v), next(nullptr), mtx(), marked(false) {}
    };

    Node* head;
    Node* tail;

    bool validate(Node* pred, Node* curr) const;
};

#endif // LIST_FINE_H
