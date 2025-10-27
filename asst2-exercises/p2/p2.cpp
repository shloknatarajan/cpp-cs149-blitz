#include <iostream>
#include <thread>
#include <string>
#include <mutex>
#include <chrono>

int global_counter = 0;

// create mutex
std::mutex mtx;

void increment_counter(int n) {
    std::lock_guard<std::mutex> lock(mtx);
    global_counter = global_counter + n;
}

void loop_increment() {
    std::lock_guard<std::mutex> lock(mtx);
    for (int i=0; i<10000000; i++) {
        global_counter += 1;
    }
}

int main() {
    const auto start_time = std::chrono::high_resolution_clock::now();
    std::thread thread1(loop_increment);
    std::thread thread2(loop_increment);
    thread1.join();
    thread2.join();
    std::cout << "Current Count: " << global_counter << std::endl;
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Elapsed Time " << elapsed_time << std::endl;

    return 0;
}