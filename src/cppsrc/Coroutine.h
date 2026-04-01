#pragma once

#include <coroutine>
#include <exception>
#include <utility>

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
