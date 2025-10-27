# Part 1 — Threads and Joining

Goal: Learn how threads start, run, and re-join the main thread.

## Hello Threads
Spawn two threads that each print “Hello from thread X”.
Have the main thread print “Hello from main”.
Observe the interleaving of output. 

## Joining vs Detaching
Create a thread that sleeps for two seconds. Try both t.join() and t.detach(); observe program exit timing.
Note what happens if you forget to join or detach, which can cause undefined behavior or a runtime error.