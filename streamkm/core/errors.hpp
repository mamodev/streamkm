#pragma once

#include <optional>
#include <format>

#include <string>
#include <cstdint>
#include <vector>

namespace streamkm {

template <typename T, typename E>
class Result {
    std::optional<T> value_;
    std::optional<E> error_;

public:
    Result(const T& value) : value_(value), error_(std::nullopt) {}
    Result(T&& value) : value_(std::move(value)), error_(std::nullopt) {}
    Result(const E& error) : value_(std::nullopt), error_(error) {}
    Result(E&& error) : value_(std::nullopt), error_(std::move(error)) {}

    inline bool is_ok() const { return value_.has_value(); }
    inline bool is_err() const { return error_.has_value(); }
    inline const E& error() const { return error_.value(); }
    inline const T& value() const { return value_.value(); }
};

template <typename E>
class Result<void, E> {
    bool ok_;
    std::optional<E> error_;

public:
    Result() : ok_(true), error_(std::nullopt) {}
    Result(const E& error) : ok_(false), error_(error) {}
    Result(E&& error) : ok_(false), error_(std::move(error)) {}

    inline bool is_ok() const { return ok_; }
    inline bool is_err() const { return !ok_; }
    inline const E& error() const { return error_.value(); }
    inline const void value() const { return; }
};

struct Error {
    uint16_t code;
    std::string desc;

    Error() : code(0), desc("unknown") {}
    explicit Error(uint16_t c) : code(c), desc("unknown") {}
    explicit Error(std::string d) : code(0), desc(std::move(d)) {}
    explicit Error(const char* d) : code(0), desc(d) {}
    Error(uint16_t c, std::string d) : code(c), desc(std::move(d)) {}

    template <typename... Args>
        requires (sizeof...(Args) > 0)
    Error(uint16_t c, const char* fmt, Args&&... args) : code(c) {
        desc = std::vformat(fmt, std::make_format_args(args...));
    }

    template <typename... Args>
        requires (sizeof...(Args) > 0)
    Error(const char* fmt, Args&&... args) : code(0) {
        desc = std::vformat(fmt, std::make_format_args(args...));
    }
};

template <typename T>
using EResult = Result<T, Error>;

#define rassert(expr, ...) \
    do { \
        if (!(expr)) { \
            return Error(__VA_ARGS__); \
        } \
    } while (0)

#define rpropagate(expr) ({ auto _res = (expr); if (_res.is_err()) return _res.error(); _res.value(); })

#if not defined(DISABLE_PASSERT)
#define passert(expr, fmt, ...) \
    do { \
        if (!(expr)) { \
            throw std::runtime_error( \
                __FILE__ ":" + std::to_string(__LINE__) + \
                " Assertion failed: " + std::format(fmt, __VA_ARGS__) \
            ); \
        } \
    } while (0)
#else
#define passert(expr, fmt, ...) do { (void)(expr); } while (0)
#endif

#define fpassert(expr, fmt, ...) \
    do { \
        if (!(expr)) { \
            throw std::runtime_error( \
                __FILE__ ":" + std::to_string(__LINE__) + \
                " Assertion failed: " + std::format(fmt, __VA_ARGS__) \
            ); \
        } \
    } while (0)

}  // namespace streamk