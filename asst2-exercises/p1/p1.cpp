#include <iostream>
#include <thread>
#include <string>

void func(std::string thread_name) {
    std::cout << "Hello from " << thread_name << std::endl;
}

int main() {
    std::thread thread1(func, "thread1");
    std::thread thread2(func, "thread2");
    thread1.join();
    thread2.join();
    func("main thread");
    return 0;
}