#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// TODO: Implement Part 6 — Producer–Consumer with condition_variable.
// Shared state suggestion:
//   std::queue<int> q; bool done = false; std::mutex m; std::condition_variable cv;
// 1) Single consumer: push 1..5, notify_one after each push; consumer waits with predicate and pops.
// 2) Multiple consumers: spawn 3 consumers; use notify_all after setting done=true so all can exit.

int main() {
    // Your code here.
    std::cout << "Part 6 placeholder. Implement producer–consumer.\n";
    return 0;
}

