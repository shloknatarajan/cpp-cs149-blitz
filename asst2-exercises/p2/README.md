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