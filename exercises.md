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


# Part 5 — Condition Variables

Goal: Coordinate threads using signals.

## Producer–Consumer
Create a shared queue, mutex, and condition variable.
The producer pushes integers one through five and calls cv.notify_one().
The consumer waits on the condition variable until data is available, pops, and prints values.

## Multiple Consumers
Spawn three consumer threads and ensure all wake correctly.
Use cv.notify_all() to broadcast completion.

## Thread Sleep and Wakeup
Start a worker thread that loops while a shared “ready” flag is false.
The main thread sleeps for two seconds, sets ready to true, and calls cv.notify_all().


# Part 6 — Mini-Project Integration

Goal: Use everything together.

## Thread Pool Prototype
Implement a small class with a vector of worker threads, a task queue protected by a mutex and condition variable, and an atomic shutdown flag.
Submit lambdas as tasks and process them concurrently.

## Benchmark Atomic versus Mutex
Measure the time required to increment a shared counter one million times using a mutex with lock_guard and then using an atomic counter.
Print or plot the timings to compare performance.