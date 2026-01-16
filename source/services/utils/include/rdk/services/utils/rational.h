/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RDK_SERVICES_UTILS_RATIONAL_H_
#define RDK_SERVICES_UTILS_RATIONAL_H_

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace rivermax
{
namespace dev_kit
{
namespace services
{

/**
 * @brief: Exception thrown by Rational class on invalid operations.
 */
class RationalException : public std::runtime_error
{
public:
    /**
     * @brief: Constructs a RationalException with an error message.
     *
     * @param [in] what: The error message.
     */
    RationalException(const std::string& what): std::runtime_error(what) {}
};

/**
 * @brief: Represents a rational number as integer + numerator/denominator.
 *
 * This class provides exact fractional arithmetic without floating-point precision loss.
 * Rational numbers are always stored in reduced form with the fractional part less than 1.
 * Only non-negative rational numbers are supported.
 */
class Rational
{
public:
    /**
     * @brief: Default constructor. Creates a Rational with value 0.
     */
    Rational();
    /**
     * @brief: Constructs a Rational from integer, numerator, and denominator.
     *
     * The value represents: integer + numerator/denominator.
     *
     * @param [in] integer: The integer part.
     * @param [in] numerator: The numerator of the fractional part.
     * @param [in] denominator: The denominator of the fractional part. Must be non-zero.
     *
     * @throws: RationalException if denominator is zero.
     */
    Rational(uint64_t integer, uint64_t numerator, uint64_t denominator);
    /**
     * @brief: Constructs a Rational from an integer value.
     *
     * @param [in] integer: The integer value.
     */
    Rational(uint64_t integer);
    /**
     * @brief: Constructs a Rational from numerator and denominator.
     *
     * The value represents: numerator/denominator.
     *
     * @param [in] numerator: The numerator.
     * @param [in] denominator: The denominator. Must be non-zero.
     *
     * @throws: RationalException if denominator is zero.
     */
    Rational(uint64_t numerator, uint64_t denominator);
    /**
     * @brief: Copy constructor.
     *
     * @param [in] other: The Rational to copy.
     */
    Rational(const Rational& other) = default;
    /**
     * @brief: Constructs a Rational from a string representation.
     *
     * Accepts formats:
     * - "numerator/denominator" (e.g., "60000/1001")
     * - "integer" (e.g., "60")
     *
     * @param [in] str: The string to parse.
     *
     * @throws: std::invalid_argument if the string format is invalid.
     * @throws: std::out_of_range if the value exceeds uint64_t range.
     * @throws: RationalException if denominator is zero.
     */
    explicit Rational(const std::string& str);
    Rational(float) = delete;
    Rational(double) = delete;
    /**
     * @brief: Constructs a Rational from any unsigned integral type.
     *
     * The fraction part is set to zero.
     *
     * @tparam T: An unsigned integral type, not uint64_t.
     * @param [in] integer: The integer value.
     */
    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value && !std::is_same<T, uint64_t>::value>::type>
    Rational(T integer) :
        m_integer(static_cast<uint64_t>(integer)),
        m_numerator(0),
        m_denominator(1)
    {
    }
    /**
     * @brief: Constructs a Rational from any signed integral type.
     *
     * The value must be non-negative. If a negative value is provided, throws RationalException.
     *
     * @tparam T: A signed integral type.
     * @param [in] integer: The integer value.
     *
     * @throws: RationalException if integer is negative.
     */
    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value>::type, typename = void>
    Rational(T integer) :
        m_integer(static_cast<uint64_t>(integer)),
        m_numerator(0),
        m_denominator(1)
    {
        if (integer < 0) {
            throw RationalException("Rational: negative rationals are not supported");
        }
    }
    /**
     * @brief: Output stream operator.
     *
     * Outputs in "{integer numerator/denominator}" format if there are both integer and fractional part,
     * otherwise returns "integer" or "numerator/denominator" format.
     *
     * @param [in] os: The output stream.
     * @param [in] other: The Rational to output.
     *
     * @return: The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Rational& other);
    /**
     * @brief: Input stream operator.
     *
     * Reads a string in "numerator/denominator" or "integer" format.
     *
     * @param [in] is: The input stream.
     * @param [out] other: The Rational to populate.
     *
     * @return: The input stream.
     */
    friend std::istream& operator>>(std::istream& is, Rational& other);
    /**
     * @brief: Copy assignment operator.
     *
     * @param [in] other: The Rational to copy.
     *
     * @return: Reference to this.
     */
    Rational& operator=(const Rational& other) = default;
    /**
     * @brief: Assignment operator from integral type.
     *
     * @tparam T: An integral type.
     * @param [in] n: The integer value to assign.
     *
     * @return: Reference to this.
     */
    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational& operator=(T n)
    {
        m_integer = n;
        m_numerator = 0;
        m_denominator = 1;

        return *this;
    }

    /** @name: Arithmetic operators.
     *  @{
     */
    Rational operator+(const Rational& other) const;
    Rational operator-(const Rational& other) const;
    Rational operator*(const Rational& other) const;
    Rational operator/(const Rational& other) const;
    Rational& operator+=(const Rational& other);
    Rational& operator-=(const Rational& other);
    Rational& operator*=(const Rational& other);
    Rational& operator/=(const Rational& other);

    Rational& operator++()
    {
        ++m_integer;
        return *this;
    }

    Rational operator++(int)
    {
        Rational tmp(*this);
        ++m_integer;
        return tmp;
    }

    Rational& operator--()
    {
        --m_integer;
        return *this;
    }

    Rational operator--(int)
    {
        Rational tmp(*this);
        --m_integer;
        return tmp;
    }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational operator+(T n) const { return *this + Rational(n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational& operator+=(T n) { return *this += Rational(n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational operator-(T n) const { return *this - Rational(n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational& operator-=(T n) { return *this -= Rational(n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational operator*(T n) const { return *this * Rational(n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational& operator*=(T n) { return *this *= Rational(n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational operator/(T n) const { return *this / Rational(n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    Rational& operator/=(T n) { return *this /= Rational(n); }
    /** @} */

    /** @name: Comparison operators.
     *  @{
     */
    bool operator==(const Rational& other) const;
    bool operator!=(const Rational& other) const;
    bool operator<(const Rational& other) const;
    bool operator>(const Rational& other) const { return other < *this; }
    bool operator<=(const Rational& other) const { return !(other < *this); }
    bool operator>=(const Rational& other) const { return other <= *this; }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator==(T n) const { return m_integer == static_cast<uint64_t>(n) && !m_numerator; }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator!=(T n) const { return !(*this == n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator<(T n) const { return m_integer < static_cast<uint64_t>(n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator<=(T n) const
    {
        uint64_t un = static_cast<uint64_t>(n);
        if (un < m_integer) return false;
        return (m_integer < un || !m_numerator);
    }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator>(T n) const { return !(*this <= n); }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool operator>=(T n) const { return !(*this < n); }
    /** @} */

    /** @name: Accessors.
     *  @{
     */
    /**
     * @brief: Returns the integer part.
     *
     * @return: The integer part.
     */
    uint64_t integer() const { return m_integer; }
    /**
     * @brief: Returns the numerator of the fractional part.
     *
     * @return: The numerator.
     */
    uint64_t numerator() const { return m_numerator; }
    /**
     * @brief: Returns the denominator of the fractional part.
     *
     * @return: The denominator.
     */
    uint64_t denominator() const { return m_denominator; }
    /**
     * @brief: Returns the total numerator (improper fraction form).
     *
     * For a value of integer + numerator/denominator, returns
     * (integer * denominator + numerator).
     *
     * @return: The total numerator.
     */
    uint64_t total_numerator() const { return m_integer * m_denominator + m_numerator; }
    /**
     * @brief: Returns the total denominator (improper fraction form).
     *
     * This is the same as denominator().
     *
     * @return: The total denominator.
     */
    uint64_t total_denominator() const { return m_denominator; }
    /** @} */

    /** @name: Conversion operators.
     *  @{
     */
    /**
     * @brief: Boolean conversion operator.
     *
     * @return: True if the value is non-zero.
     */
    explicit operator bool() const { return m_integer || m_numerator; }
    /**
     * @brief: Integral conversion operator.
     *
     * @tparam T: An integral type.
     * @return: Integer part converted to the output type.
     */
    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    explicit operator T() const
    {
        return static_cast<T>(integer());
    }
    /**
     * @brief: String conversion operator.
     *
     * Returns "{integer numerator/denominator}" format if there are both integer and fractional part,
     * otherwise returns "integer" or "numerator/denominator" format.
     *
     * @return: The string representation.
     */
    operator std::string() const;
    /**
     * @brief: Converts the Rational to a string.
     *
     * Returns "{integer numerator/denominator}" format if there are both integer and fractional part,
     * otherwise returns "integer" or "numerator/denominator" format.
     *
     * @return: The string representation.
     */
     std::string to_string() const;
    /**
     * @brief: Converts the Rational to a string with total numerator and denominator (improper fraction form).
     *
     * Returns "numerator/denominator" format if there's a fractional part,
     * otherwise returns "integer" format.
     *
     * @return: The string representation.
     * @note: Calculating the total numerator may overflow for large values.
     */
    std::string to_total_string() const;
    /** @} */

private:
    /**
     * @brief: Computes the greatest common divisor of two integers.
     *
     * @param [in] i1: The first integer.
     * @param [in] i2: The second integer.
     *
     * @return: The greatest common divisor.
     */
    static uint64_t gcd(uint64_t i1, uint64_t i2);
    /**
     * @brief: Computes the least common denominator of two integers.
     *
     * @param [in] i1: The first integer.
     * @param [in] i2: The second integer.
     *
     * @return: The least common denominator.
     */
    static uint64_t lcd(uint64_t i1, uint64_t i2);
    /**
     * @brief: Reduces two integers by their greatest common divisor.
     *
     * @param [in,out] i1: The first integer, modified in place.
     * @param [in,out] i2: The second integer, modified in place.
     *
     * @return: The greatest common divisor used for reduction.
     */
    static uint64_t reduce_two(uint64_t& i1, uint64_t& i2);
    /**
     * @brief: Reduces two integers by their greatest common divisor.
     *
     * @param [in,out] i1: The first integer, modified in place.
     * @param [in,out] i2: The second integer, modified in place.
     */
    static void reduce(uint64_t& i1, uint64_t& i2);
    /**
     * @brief: Initializes the Rational from components.
     *
     * Normalizes the value so the fractional part is less than 1 and reduced.
     *
     * @param [in] integer: The integer part.
     * @param [in] numerator: The numerator of the fractional part.
     * @param [in] denominator: The denominator of the fractional part.
     */
    void init(uint64_t integer, uint64_t numerator, uint64_t denominator);
    /**
     * @brief: Initializes the Rational by normalizing the current components.
     *
     * Reduces the fraction and ensures the fractional part is less than 1.
     */
    void init();
    /**
     * @brief: Performs multiplication or division of two Rationals.
     *
     * @param [in] lhs: The left-hand side operand.
     * @param [in] rhs: The right-hand side operand.
     * @param [in] is_multiply: True for multiplication, false for division.
     *
     * @return: The result of the operation.
     */
    Rational mul_div(const Rational& lhs, const Rational& rhs, bool is_multiply) const;
    /**
     * @brief: Performs multiplication or division and assigns the result.
     *
     * @param [in] lhs: The left-hand side operand.
     * @param [in] rhs: The right-hand side operand.
     * @param [in] is_multiply: True for multiplication, false for division.
     *
     * @return: Reference to this.
     */
    Rational& mul_div_assign(const Rational& lhs, const Rational& rhs, bool is_multiply);
    /**
     * @brief: Performs addition or subtraction of two Rationals.
     *
     * @param [in] lhs: The left-hand side operand.
     * @param [in] rhs: The right-hand side operand.
     * @param [in] is_add: True for addition, false for subtraction.
     *
     * @return: The result of the operation.
     */
    Rational add_sub(const Rational& lhs, const Rational& rhs, bool is_add) const;
    /**
     * @brief: Performs addition or subtraction and assigns the result.
     *
     * @param [in] other: The other operand.
     * @param [in] is_add: True for addition, false for subtraction.
     *
     * @return: Reference to this.
     */
    Rational& add_sub_assign(const Rational& other, bool is_add);

    uint64_t m_integer;
    uint64_t m_numerator;
    uint64_t m_denominator;
};

/**
 * @brief: Casts a Rational to a numeric type.
 *
 * @tparam T: The target numeric type.
 * @param [in] a: The Rational to cast.
 *
 * @return: The value as the target type.
 */
template <typename T>
inline T rational_cast(const Rational& a)
{
    T ret = static_cast<T>(a.numerator());
    ret /= static_cast<T>(a.denominator());
    ret += static_cast<T>(a.integer());
    return ret;
}

/** @name: Free function operators for integral types on the left.
 *  @{
 */
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
Rational operator+(T lhs, const Rational& rhs) { return rhs + lhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
Rational operator-(T lhs, const Rational& rhs) { return Rational(lhs) - rhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
Rational operator*(T lhs, const Rational& rhs) { return rhs * lhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
Rational operator/(T lhs, const Rational& rhs) { return Rational(lhs) / rhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool operator<(T lhs, const Rational& rhs) { return rhs > lhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool operator<=(T lhs, const Rational& rhs) { return rhs >= lhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool operator>(T lhs, const Rational& rhs) { return rhs < lhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool operator>=(T lhs, const Rational& rhs) { return rhs <= lhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool operator==(T lhs, const Rational& rhs) { return rhs == lhs; }

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool operator!=(T lhs, const Rational& rhs) { return rhs != lhs; }
/** @} */

} // namespace services
} // namespace dev_kit
} // namespace rivermax

namespace std {
inline string to_string(const rivermax::dev_kit::services::Rational& r)
{
    return r.to_string();
}

template<>
struct hash<rivermax::dev_kit::services::Rational>
{
    size_t operator()(const rivermax::dev_kit::services::Rational& r) const noexcept
    {
        return hash<uint64_t>()(r.total_numerator()) ^
               (hash<uint64_t>()(r.total_denominator()) << 1);
    }
};
}

template<>
struct std::common_type<rivermax::dev_kit::services::Rational, rivermax::dev_kit::services::Rational> {
    using type = rivermax::dev_kit::services::Rational;
};

template<typename T>
struct std::common_type<T, rivermax::dev_kit::services::Rational> {
    using type = rivermax::dev_kit::services::Rational;
};

template<typename T>
struct std::common_type<rivermax::dev_kit::services::Rational, T> {
    using type = rivermax::dev_kit::services::Rational;
};

#endif // RDK_SERVICES_UTILS_RATIONAL_H_
