#include "list_fine.h"
#include <limits>
#include <cassert>

FineList::FineList() {
    head = new Node(INT_MIN);
    tail = new Node(INT_MAX);
    head->next = tail;
}

FineList::~FineList() {
    Node* cur = head;
    while (cur) {
        Node* tmp = cur->next;
        delete cur;
        cur = tmp;
    }
}

void FineList::locate(int value, Node*& pred, Node*& curr) const {
    pred = head;
    curr = pred->next;
    // traverse until curr->value >= value
    while (curr->value < value) {
        pred = curr;
        curr = curr->next;
    }
}

// validate that pred and curr are still adjacent and not marked
bool FineList::validate(Node* pred, Node* curr) {
    // pred and curr must not be marked and pred->next must still point to curr
    if (pred->marked.load(std::memory_order_acquire)) return false;
    if (curr->marked.load(std::memory_order_acquire)) return false;
    return pred->next == curr;
}

bool FineList::find(int value) const {
    Node* pred;
    Node* curr;
    locate(value, pred, curr);
    // present if curr->value == value and not logically removed
    return (curr->value == value) && (!curr->marked.load(std::memory_order_acquire));
}

bool FineList::insert(int value) {
    while (true) {
        Node* pred;
        Node* curr;
        locate(value, pred, curr);

        // acquire locks on pred and curr in address order to avoid deadlock
        // but classical algorithm locks pred then curr (pred->mtx, curr->mtx)
        std::unique_lock<std::mutex> lpred(pred->mtx, std::defer_lock);
        std::unique_lock<std::mutex> lcurr(curr->mtx, std::defer_lock);
        std::lock(pred->mtx, curr->mtx);
        // adopt locks
        lpred.release(); lcurr.release();
        std::unique_lock<std::mutex> upred(pred->mtx, std::adopt_lock);
        std::unique_lock<std::mutex> ucurr(curr->mtx, std::adopt_lock);

        // validate
        if (!validate(pred, curr)) {
            // unlock and retry
            // unique_lock destructors will unlock
            continue;
        }

        if (curr->value == value) {
            // already present
            return false;
        } else {
            Node* node = new Node(value);
            node->next = curr;
            pred->next = node;
            return true;
        }
    }
}

bool FineList::remove(int value) {
    while (true) {
        Node* pred;
        Node* curr;
        locate(value, pred, curr);

        std::unique_lock<std::mutex> lpred(pred->mtx, std::defer_lock);
        std::unique_lock<std::mutex> lcurr(curr->mtx, std::defer_lock);
        std::lock(pred->mtx, curr->mtx);
        lpred.release(); lcurr.release();
        std::unique_lock<std::mutex> upred(pred->mtx, std::adopt_lock);
        std::unique_lock<std::mutex> ucurr(curr->mtx, std::adopt_lock);

        if (!validate(pred, curr)) {
            continue;
        }

        if (curr->value != value) {
            return false; // not present
        } else {
            // logical removal
            curr->marked.store(true, std::memory_order_release);
            // physical removal
            pred->next = curr->next;
            // it's safe to delete curr now because we hold locks on pred and curr
            delete curr;
            return true;
        }
    }
}

