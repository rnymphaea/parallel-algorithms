#include "list_fine.h"
#include <limits>

FineList::FineList() {
    head = new Node(0);
    tail = new Node(0);
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

bool FineList::validate(Node* pred, Node* curr) const {
    if (pred->marked.load(std::memory_order_acquire)) return false;
    if (curr->marked.load(std::memory_order_acquire)) return false;
    return pred->next == curr;
}

bool FineList::find(int value) const {
    Node* curr = head->next;
    while (curr != tail) {
        if (curr->value == value && !curr->marked.load(std::memory_order_acquire)) {
            return true;
        }
        curr = curr->next;
    }
    return false;
}

bool FineList::insert(int value) {
    while (true) {
        Node* pred = head;
        Node* curr = head->next;
        
        while (curr != tail) {
            pred = curr;
            curr = curr->next;
        }

        std::unique_lock<std::mutex> lock_pred(pred->mtx);
        std::unique_lock<std::mutex> lock_curr(curr->mtx);

        if (!validate(pred, curr)) {
            continue;
        }

        Node* newNode = new Node(value);
        newNode->next = tail;
        pred->next = newNode;
        return true;
    }
}

bool FineList::remove(int value) {
    while (true) {
        Node* pred = head;
        Node* curr = head->next;

        while (curr != tail && curr->value != value) {
            pred = curr;
            curr = curr->next;
        }

        if (curr == tail) {
            return false;
        }

        std::unique_lock<std::mutex> lock_pred(pred->mtx);
        std::unique_lock<std::mutex> lock_curr(curr->mtx);

        if (!validate(pred, curr)) {
            continue;
        }

        if (curr->value != value) {
            return false;
        }

        curr->marked.store(true, std::memory_order_release);
        pred->next = curr->next;
        delete curr;
        return true;
    }
}
