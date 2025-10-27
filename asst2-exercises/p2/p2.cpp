#include <iostream>
#include <thread>
#include <string>
#include <mutex>

int global_counter = 0;

// create mutex
std::mutex mtx;

void increment_counter(int n) {
    std::lock_guard<std::mutex> lock(mtx);
    global_counter = global_counter + n;
}

int main() {
    std::thread thread1(increment_counter, 1);
    std::thread thread2(increment_counter, 2);
    thread1.join();
    thread2.join();
    increment_counter(3);
    std::cout << "Current Count: " << global_counter << std::endl;

    return 0;
}