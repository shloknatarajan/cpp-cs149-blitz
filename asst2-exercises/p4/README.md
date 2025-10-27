# Part 4 — Lock Guards and Unique Locks

Goal: Practice RAII locking and manual control.

## Scope Unlock
Inside a function, use std::lock_guard and purposely throw an exception.
Observe that the mutex is still released and no deadlock occurs.

## Manual Locking
Use std::unique_lock but delay calling .lock() until later in the function.
Unlock early with .unlock() and continue other work safely.