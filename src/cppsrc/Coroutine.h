#pragma once

#include <coroutine>
#include <exception>
#include <utility>

// ---------------------------------------------------------------------------
// Per-thread coroutine frame pool
//
// Overriding operator new/delete in promise_type routes every frame through
// this pool instead of the global heap.  Freed frames are linked through
// their own memory (the first word becomes a next pointer), so allocation is
// a single pointer pop and deallocation is a single pointer push — no locks,
// no virtual dispatch, no fragmentation.
//
// All rollout() frames are the same size (same function, same locals), so
// one list covers the entire workload.  Falls back to ::operator new only
// on the first allocation of each distinct size.
// ---------------------------------------------------------------------------

#include <vector>

struct CoroFramePool {
    struct Node {
        Node *next;
    };
    Node *head = nullptr;

    std::size_t frameSize = 0;
    std::vector<void *> allocatedChunks;

    // Allocate 1024 frames per OS trip to guarantee cache locality
    static constexpr int CHUNK_SIZE = 1024;

    void *allocate(std::size_t n) {
        // Lock the pool to the size of the first frame it sees
        if (frameSize == 0)
            frameSize = n;

        // Safety: If a different sized coroutine uses this pool, route to OS
        // heap
        if (n != frameSize)
            return ::operator new(n);

        // Fast Path: Pop from free list
        if (head) {
            Node *p = head;
            head = head->next;
            return p;
        }

        // Slow Path (warmed up fast): Allocate a large contiguous slab
        char *chunk = static_cast<char *>(::operator new(CHUNK_SIZE * n));
        allocatedChunks.push_back(chunk);

        // Chop the slab into frames and link them
        for (int i = 1; i < CHUNK_SIZE - 1; ++i) {
            Node *node = reinterpret_cast<Node *>(chunk + i * n);
            node->next = reinterpret_cast<Node *>(chunk + (i + 1) * n);
        }
        reinterpret_cast<Node *>(chunk + (CHUNK_SIZE - 1) * n)->next = nullptr;

        head = reinterpret_cast<Node *>(chunk + n);
        return chunk;
    }

    void deallocate(void *p, std::size_t n) noexcept {
        if (n != frameSize) {
            ::operator delete(p);
            return;
        }
        // Push the reclaimed frame back onto the head of the free list
        auto *node = static_cast<Node *>(p);
        node->next = head;
        head = node;
    }

    ~CoroFramePool() {
        // Free the massive contiguous chunks all at once
        for (void *chunk : allocatedChunks) {
            ::operator delete(chunk);
        }
    }
};

inline thread_local CoroFramePool coroPool;

// FinalAwaiter: on co_return, symmetric-transfer to the parent that is
// co_await-ing this task, or fall through to noop if this is a root task.
template <typename Promise> struct FinalAwaiter {
    bool await_ready() noexcept { return false; }
    std::coroutine_handle<>
    await_suspend(std::coroutine_handle<Promise> h) noexcept {
        auto cont = h.promise().continuation_;
        return cont ? cont : std::noop_coroutine();
    }
    void await_resume() noexcept {}
};

template <typename T> class Task {
  public:
    struct promise_type {
        T value_{};
        std::exception_ptr exception_{};
        std::coroutine_handle<> continuation_{};

        static void *operator new(std::size_t n) {
            return coroPool.allocate(n);
        }
        static void operator delete(void *p, std::size_t n) noexcept {
            coroPool.deallocate(p, n);
        }

        Task get_return_object() noexcept {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        FinalAwaiter<promise_type> final_suspend() noexcept { return {}; }
        void return_value(T v) noexcept { value_ = std::move(v); }
        void unhandled_exception() noexcept {
            exception_ = std::current_exception();
        }
    };

    explicit Task(std::coroutine_handle<promise_type> h) noexcept
        : handle_(h) {}
    Task(Task &&o) noexcept : handle_(std::exchange(o.handle_, {})) {}
    Task(const Task &) = delete;
    Task &operator=(Task &&) = delete;
    Task &operator=(const Task &) = delete;
    ~Task() {
        if (handle_)
            handle_.destroy();
    }

    bool done() const noexcept { return !handle_ || handle_.done(); }
    std::coroutine_handle<promise_type> handle() const noexcept {
        return handle_;
    }

    // Awaitable — symmetric transfer directly into child, bypassing the
    // scheduler.
    bool await_ready() const noexcept { return done(); }
    std::coroutine_handle<>
    await_suspend(std::coroutine_handle<> caller) noexcept {
        handle_.promise().continuation_ = caller;
        return handle_;
    }
    T await_resume() {
        if (handle_.promise().exception_)
            std::rethrow_exception(handle_.promise().exception_);
        return std::move(handle_.promise().value_);
    }

  private:
    std::coroutine_handle<promise_type> handle_;
};
