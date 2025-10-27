#include <condition_variable>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

// TODO: Implement Part 5 — Ready flag using condition_variable.
// Steps:
// 1) Create shared state: bool ready = false; std::mutex m; std::condition_variable cv;
// 2) Start a worker thread that waits until ready becomes true using cv.wait with a predicate.
// 3) In main, sleep ~2 seconds, then set ready to true under the lock and notify_all.


int main() {
    // Your code here.
    std::cout << "Part 5 placeholder. Implement the ready-flag demo.\n";
    return 0;
}

