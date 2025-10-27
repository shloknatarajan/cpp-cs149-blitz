# Part 3 — Atomics

Goal: Understand atomic versus locked increments.

## Atomic Counter
Replace the mutex with std::atomic counter initialized to zero.
Compare runtime speed and correctness with the mutex-based version.

## Atomic Flag
Implement a simple “once” flag where multiple threads attempt to set an atomic boolean.
Only the first one succeeds and prints “First!”.
