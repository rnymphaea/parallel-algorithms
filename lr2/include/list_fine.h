#ifndef LIST_FINE_H
#define LIST_FINE_H

#include <mutex>
#include <atomic>
#include <climits>

// Fine-grained (lazy) sorted singly-linked list for `int` keys.
// - sentinel head with INT_MIN and tail with INT_MAX
// - find/contains is lock-free (no mutex taken), uses atomic 'marked' to ignore logically removed nodes
// - insert/remove lock only two adjacent nodes (pred and curr) and validate
// This is the classical "lazy list" fine-grained implementation.

class FineList {
public:
    FineList();
    ~FineList();

    // Insert value if not present. Returns true if inserted.
    bool insert(int value);

    // Delete value if present. Returns true if deleted.
    bool remove(int value);

    // Find value (returns true if present and not logically removed).
    bool find(int value) const;

private:
    struct Node {
        int value;
        Node* next;
        std::mutex mtx;
        std::atomic<bool> marked; // logical removal flag

        Node(int v) : value(v), next(nullptr), mtx(), marked(false) {}
    };

    Node* head; // points to sentinel INT_MIN
    Node* tail; // points to sentinel INT_MAX

    // helper: locate pred and curr such that pred->value < value <= curr->value
    // This function does traversal without acquiring locks.
    void locate(int value, Node*& pred, Node*& curr) const;

    // validate that pred and curr are adjacent, not marked, and pred->next == curr
    static bool validate(Node* pred, Node* curr);
};

#endif // LIST_FINE_H

