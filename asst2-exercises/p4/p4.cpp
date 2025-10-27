#include <thread>
#include <iostream>
#include <mutex>
#include <vector>

int counter = 0;
std::mutex mtx;

void increment_counter() {
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
    lock.lock();
    for (int i = 0; i < 1000000; i++) { 
        counter++;
    }
    lock.unlock();
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back(increment_counter);
    }

    for (auto& t: threads) {
        t.join();
    }

    std::cout << "Final Count: " << counter << std::endl;

    return 0;
}