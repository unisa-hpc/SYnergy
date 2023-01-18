#ifndef SYNERGY_UTILS_H
#define SYNERGY_UTILS_H

namespace details {

// --------------- Check position of a type in variadic template ----------------

template <typename T, typename... Args>
struct Index;

template <typename T, typename... Args>
struct Index<T, T, Args...> : std::integral_constant<std::size_t, 0> {};

template <typename T, typename U, typename... Args>
struct Index<T, U, Args...> : std::integral_constant<std::size_t, 1 + Index<T, Args...>::value> {};

// --------------- Check if a type is present in varidic template ----------------

template <typename T, typename... Args>
struct is_present;

template <typename T>
struct is_present<T> : std::false_type {};

template <typename T, typename... Args>
struct is_present<T, T, Args...> : std::true_type {};

template <typename T, typename U, typename... Args>
struct is_present<T, U, Args...> : is_present<T, Args...> {};

template <typename T, typename... Args>
inline constexpr bool is_present_v = is_present<T, Args...>::value;

// --------------- Remove last element from a tuple ----------------

template <class... Args, std::size_t... Is>
constexpr auto remove_last_impl(std::tuple<Args...> tp, std::index_sequence<Is...>)
{
  return std::tuple{std::get<Is>(tp)...};
}

template <class... Args>
constexpr auto remove_last(std::tuple<Args...> tp)
{
  return remove_last_impl(tp, std::make_index_sequence<sizeof...(Args) - 1>{});
}

template <class... Args>
constexpr auto has_property()
{
  return (sycl::is_property_v<Args> || ...);
}

template <class... Args>
inline constexpr bool has_property_v = has_property<Args...>();

} // namespace details
#endif // SYNERGY_UTILS_H