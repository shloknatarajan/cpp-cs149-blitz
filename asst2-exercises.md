# Part 1 — Threads and Joining

Goal: Learn how threads start, run, and re-join the main thread.

## Hello Threads
Spawn two threads that each print “Hello from thread X”.
Have the main thread print “Hello from main”.
Observe the interleaving of output. 

## Joining vs Detaching
Create a thread that sleeps for two seconds. Try both t.join() and t.detach(); observe program exit timing.
Note what happens if you forget to join or detach, which can cause undefined behavior or a runtime error.

# Part 2 — Mutex and Race Conditions

Goal: Observe race conditions and fix them.

## Race Condition Demo
Create a shared integer counter initialized to zero.
Spawn four threads, each incrementing the counter ten thousand times.
Print the final value; it should not equal forty thousand due to race conditions.

## Fix with std::mutex
Add a global std::mutex and surround the increment with m.lock() and m.unlock().
Confirm that the result is always forty thousand.

## Fix with std::lock_guard
Replace manual locking and unlocking with std::lock_guard std::mutex guard(m).
Verify correctness and note the shorter, safer code.


# Part 3 — Atomics

Goal: Understand atomic versus locked increments.

## Atomic Counter
Replace the mutex with std::atomic counter initialized to zero.
Compare runtime speed and correctness with the mutex-based version.

## Atomic Flag
Implement a simple “once” flag where multiple threads attempt to set an atomic boolean.
Only the first one succeeds and prints “First!”.


# Part 4 — Lock Guards and Unique Locks

Goal: Practice RAII locking and manual control.

## Scope Unlock
Inside a function, use std::lock_guard and purposely throw an exception.
Observe that the mutex is still released and no deadlock occurs.

## Manual Locking
Use std::unique_lock but delay calling .lock() until later in the function.
Unlock early with .unlock() and continue other work safely.


# Part 5 — Condition Variable Basics

Goal: Learn how to wait and signal safely.

What you’ll learn:
- What `std::condition_variable` is and why we use it.
- Why `std::unique_lock<std::mutex>` is required for `wait`.
- How to use the predicate form of `wait` to avoid spurious wakeups.
- The difference between `notify_one` and `notify_all` and when to use each.

## Minimal "Ready" Flag
- Shared state: `bool ready = false; std::mutex m; std::condition_variable cv;`
- Worker thread: acquires `std::unique_lock<std::mutex> lock(m)`, then calls `cv.wait(lock, [&]{ return ready; });` and prints "ready!" once the predicate is true.
- Main thread: sleep for ~2 seconds, then under the same mutex set `ready = true;` and call `cv.notify_all();`.

Checklist and gotchas:
- Always use `std::unique_lock` (not `lock_guard`) with `cv.wait`.
- Prefer `cv.wait(lock, predicate)` over manual `while(!pred) cv.wait(lock);` but both are correct.
- Guard both the predicate and any shared data with the same mutex.
- Only use `if` for one-shot checks; use a loop or predicate with `wait` to handle spurious wakeups.

## Sleep and Wakeup
Start a worker that loops until a shared `ready` flag becomes true, using `cv.wait` with a predicate.
Have main sleep for two seconds, set `ready = true` under the lock, then `cv.notify_all()` to wake the worker.


# Part 6 — Producer–Consumer Pattern

Goal: Apply condition variables to coordinate producers and consumers.

## Single Consumer
- Shared state: `std::queue<int> q; bool done = false; std::mutex m; std::condition_variable cv;`
- Producer: pushes numbers 1..5, each time: lock, push, unlock, then `cv.notify_one()`.
- Consumer: waits with `cv.wait(lock, [&]{ return !q.empty() || done; });`, pops when `!q.empty()`, processes, repeats. Exits when `done && q.empty()`.
- After producing, set `done = true;` under the lock and `cv.notify_all()` so the consumer can exit.

## Multiple Consumers
- Spawn three consumers using the same `q`, `m`, `cv`, and `done`.
- Use `cv.notify_all()` when setting `done = true;` so all sleepers wake and exit.
- Ensure all consumers join cleanly and there is no busy-waiting.

Notes and tips:
- Use `notify_one` for waking a single waiting consumer after pushing one item.
- Use `notify_all` to broadcast a state change (e.g., shutdown via `done = true`).
- Always re-check the condition after waking; multiple threads may contend to pop.

Where to implement:
- See `asst2-exercises/p5a` for Part 5A (basics) and `asst2-exercises/p5b` for Part 5B (producer–consumer) with starter files and step-by-step tasks.


# Part 7 — Mini-Project Integration

Goal: Use everything together.

## Thread Pool Prototype
Implement a small class with a vector of worker threads, a task queue protected by a mutex and condition variable, and an atomic shutdown flag.
Submit lambdas as tasks and process them concurrently.

## Benchmark Atomic versus Mutex
Measure the time required to increment a shared counter one million times using a mutex with lock_guard and then using an atomic counter.
Print or plot the timings to compare performance.
