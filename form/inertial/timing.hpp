#include <cstdint>
#include <string>

// TODO: This is all borrowed from evalio, just import from their or include copy?

namespace form {

struct Duration {
  // Also tried saving this in seconds, but found we had occasional floating
  // point errors when adding/subtracting durations.
  int64_t nsec;

  static Duration from_sec(double sec) {
    return Duration{.nsec = int64_t(sec * 1e9)};
  }

  static Duration from_nsec(int64_t nsec) { return Duration{.nsec = nsec}; }

  double to_sec() const { return double(nsec) / 1e9; }

  int64_t to_nsec() const { return nsec; }

  std::string toString() const { return "Duration(" + toStringBrief() + ")"; }

  std::string toStringBrief() const { return std::to_string(to_sec()); }

  bool operator<(const Duration &other) const { return nsec < other.nsec; }

  bool operator>(const Duration &other) const { return nsec > other.nsec; }

  bool operator==(const Duration &other) const { return nsec == other.nsec; }

  bool operator!=(const Duration &other) const { return !(*this == other); }

  Duration operator-(const Duration &other) const {
    return Duration::from_nsec(nsec - other.nsec);
  }

  Duration operator+(const Duration &other) const {
    return Duration::from_nsec(nsec + other.nsec);
  }
};

struct Stamp {
  uint32_t sec;
  uint32_t nsec;

  static Stamp from_sec(double sec) {
    return Stamp{.sec = uint32_t(sec),
                 .nsec = uint32_t((sec - uint32_t(sec)) * 1e9)};
  }

  static Stamp from_nsec(uint64_t nsec) {
    return Stamp{.sec = uint32_t(nsec / 1e9),
                 .nsec = uint32_t(nsec % uint64_t(1e9))};
  }

  uint64_t to_nsec() const { return uint64_t(sec) * uint64_t(1e9) + nsec; }

  double to_sec() const { return double(sec) + double(nsec) * 1e-9; }

  std::string toString() const { return "Stamp(" + toStringBrief() + ")"; }

  std::string toStringBrief() const {
    size_t n_zeros = 9;
    auto nsec_str = std::to_string(nsec);
    auto nsec_str_leading =
        std::string(9 - std::min(n_zeros, nsec_str.length()), '0') + nsec_str;
    return std::to_string(sec) + "." + nsec_str_leading;
  }

  bool operator<(const Stamp &other) const {
    return sec < other.sec || (sec == other.sec && nsec < other.nsec);
  }

  bool operator>(const Stamp &other) const {
    return sec > other.sec || (sec == other.sec && nsec > other.nsec);
  }

  bool operator==(const Stamp &other) const {
    return sec == other.sec && nsec == other.nsec;
  }

  bool operator!=(const Stamp &other) const { return !(*this == other); }

  Stamp operator-(const Duration &other) const {
    return Stamp::from_nsec(to_nsec() - other.nsec);
  }

  Stamp operator+(const Duration &other) const {
    return Stamp::from_nsec(to_nsec() + other.nsec);
  }

  Duration operator-(const Stamp &other) const {
    return Duration::from_sec(to_sec() - other.to_sec());
  }
};

} // namespace form