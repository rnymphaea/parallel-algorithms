#include "list_coarse.h"
#include <cstdlib>

CoarseList::CoarseList(): head(nullptr) {}

CoarseList::~CoarseList() {
    std::lock_guard<std::mutex> lg(mtx);
    Node* cur = head;
    while (cur) {
        Node* tmp = cur;
        cur = cur->next;
        delete tmp;
    }
}

bool CoarseList::insert(int value) {
    std::lock_guard<std::mutex> lg(mtx);
    Node** curp = &head;
    while (*curp) {
        if ((*curp)->value == value) return false;
        curp = &((*curp)->next);
    }
    Node* node = new Node(value);
    *curp = node;
    return true;
}

bool CoarseList::remove(int value) {
    std::lock_guard<std::mutex> lg(mtx);
    Node** curp = &head;
    while (*curp) {
        if ((*curp)->value == value) {
            Node* to_del = *curp;
            *curp = to_del->next;
            delete to_del;
            return true;
        }
        curp = &((*curp)->next);
    }
    return false;
}

bool CoarseList::find(int value) {
    std::lock_guard<std::mutex> lg(mtx);
    Node* cur = head;
    while (cur) {
        if (cur->value == value) return true;
        cur = cur->next;
    }
    return false;
}
