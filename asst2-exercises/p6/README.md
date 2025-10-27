# Part 5B — Producer–Consumer Pattern

Goal: Use a condition variable to safely coordinate producers and consumers.

Design:
- Shared state: `std::queue<int> q; bool done = false; std::mutex m; std::condition_variable cv;`
- Producer pushes work items and wakes a consumer with `notify_one`.
- Consumers wait until there is data or `done`, then pop and process.

Step-by-step:
1) Single consumer
   - Producer: push integers 1..5. For each item: lock `m`, `q.push(i)`, unlock, then `cv.notify_one()`.
   - Consumer: `std::unique_lock<std::mutex> lock(m);`
     `cv.wait(lock, [&]{ return !q.empty() || done; });`
     If `!q.empty()`, pop and process; if `done && q.empty()`, exit.
   - After producing, under the lock set `done = true;` then `cv.notify_all()`.

2) Multiple consumers
   - Spawn three identical consumers; all share `q`, `m`, `cv`, `done`.
   - Continue to use `notify_one()` for each push; use `notify_all()` when setting `done = true;`.
   - Join all consumers cleanly.

3) Discussion / pitfalls
   - Always use the same mutex for both the condition and the queue operations.
   - Re-check the condition after wake; multiple consumers may race to pop.
   - Avoid holding the lock while doing slow work; pop first, then unlock and process.

Implement your solution in `p5b.cpp`. Start with the single-consumer version, then extend to multiple consumers.

