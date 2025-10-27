#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <string>

std::atomic<int> counter(0);
std::atomic<bool> flag(false);

void increment_atomic() {
    for (int i = 0; i < 1000000; i++) {
        counter++;
    }
}

void set_flag_true() {
    std::thread::id current_thread_id = std::this_thread::get_id();
    flag = true;
    std::cout << "Winning Thread ID: " << current_thread_id << std::endl;
}

int main() {
    // Create the threads
    std::vector<std::thread> threads;
    // Create 5 threads that increment
    for (int i = 0; i < 5; i++) {
        threads.emplace_back(increment_atomic);
    }

    // Join all the threads
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final Count: " << counter << std::endl;

    std::thread t1(set_flag_true);
    t1.join();
    return 0;
}
