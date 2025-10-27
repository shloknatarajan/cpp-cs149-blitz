# Part 5A — Condition Variable Basics

Goal: Learn how to wait and signal safely.

You’ll learn:
- What `std::condition_variable` does: allows a thread to sleep until a condition becomes true.
- Why `std::unique_lock<std::mutex>` is required: `wait` needs an unlockable lock.
- How to use the predicate overload of `wait` to handle spurious wakeups.
- When to use `notify_one` vs `notify_all`.

Mental model:
- A condition variable does not store state. You store state in your own variables guarded by a mutex (e.g., `ready`, `queue` size).
- The typical pattern is: lock -> check condition; if false, `cv.wait(lock, predicate)`; after wake, re-check the condition; then proceed.

Starter task: Ready flag
1) Shared data: `bool ready = false; std::mutex m; std::condition_variable cv;`
2) Worker thread:
   - `std::unique_lock<std::mutex> lock(m);`
   - `cv.wait(lock, [&]{ return ready; });`
   - Print: "ready!"
3) Main thread:
   - Sleep ~2 seconds.
   - `std::lock_guard<std::mutex> g(m); ready = true;`
   - `cv.notify_all();`

Gotchas:
- Always guard the condition and data with the same mutex.
- Prefer `cv.wait(lock, predicate)` over manual while-loops; both are acceptable.
- Do not busy-wait; `wait` releases the mutex while sleeping and re-acquires it before returning.

Extension: Timeout wait
- Try `cv.wait_for(lock, 500ms, [&]{ return ready; })` and report whether you timed out or saw `ready`.

Implement your solution in `p5a.cpp` and run it to observe the sleep/wake behavior.

