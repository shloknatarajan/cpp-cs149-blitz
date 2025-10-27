#include <iostream>
#include <thread>
#include <string>
#include <chrono>

void func(std::string thread_name) {
    std::cout << "Hello from " << thread_name << std::endl;
}

void sleep_for_n(int n) {
    std::this_thread::sleep_for(std::chrono::seconds(n));
}

int main() {
    const auto start_time = std::chrono::high_resolution_clock::now();
    std::thread thread1(func, "thread1");
    std::thread thread2([]() {
        sleep_for_n(2);
        func("thread2");
    });
    thread1.join();
    thread2.join();
    const auto end_time = std::chrono::high_resolution_clock::now();
    func("main thread");
    const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Total Time: " << elapsed_time << std::endl;
    return 0;
}