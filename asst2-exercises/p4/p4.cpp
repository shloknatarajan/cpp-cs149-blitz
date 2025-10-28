#include <thread>
#include <mutex>
#include <iostream>
#include <vector>

/*
Create a counter
have locks and unlocks around it using unique lock instead though
*/

std::mutex mtx;
int counter = 0;

void increment_counter() {
    std::unique_lock<std::mutex> ulock(mtx, std::defer_lock);
    ulock.lock();
    for (int i = 0; i < 1000000; i++) {
        counter++;
    }
    ulock.unlock();
}

int main() {
    // Create threads
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; i++) {
        threads.emplace_back(increment_counter);
    }

    for (auto& t: threads) {
        t.join();
    }
    std::cout << "Final Count: " << counter << std::endl;
    return 0;
}