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

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "rdk/services/utils/rational.h"

namespace rivermax
{
namespace dev_kit
{
namespace services
{

static_assert(!std::is_convertible<Rational, bool>::value,
    "Rational should not be implicitly convertible to bool");
static_assert(!std::is_convertible<Rational, float>::value,
    "Rational should not be implicitly convertible to float");
static_assert(!std::is_convertible<Rational, double>::value,
    "Rational should not be implicitly convertible to double");
static_assert(!std::is_convertible<Rational, uint8_t>::value,
    "Rational should not be implicitly convertible to uint8_t");
static_assert(!std::is_convertible<Rational, int8_t>::value,
    "Rational should not be implicitly convertible to int8_t");
static_assert(!std::is_convertible<Rational, uint16_t>::value,
    "Rational should not be implicitly convertible to uint16_t");
static_assert(!std::is_convertible<Rational, int16_t>::value,
    "Rational should not be implicitly convertible to int16_t");
static_assert(!std::is_convertible<Rational, uint32_t>::value,
    "Rational should not be implicitly convertible to uint32_t");
static_assert(!std::is_convertible<Rational, int32_t>::value,
    "Rational should not be implicitly convertible to int32_t");
static_assert(!std::is_convertible<Rational, uint64_t>::value,
    "Rational should not be implicitly convertible to uint64_t");
static_assert(!std::is_convertible<Rational, int64_t>::value,
    "Rational should not be implicitly convertible to int64_t");

Rational::Rational(uint64_t integer, uint64_t numerator, uint64_t denominator)
{
    init(integer, numerator, denominator);
}

Rational::Rational() :
    m_integer(0),
    m_numerator(0),
    m_denominator(1)
{
}

Rational::Rational(uint64_t integer) :
    m_integer(integer),
    m_numerator(0),
    m_denominator(1)
{
}

Rational::Rational(uint64_t numerator, uint64_t denominator)
    : Rational(0, numerator, denominator)
{
}

Rational::Rational(const std::string& str)
{
    auto is_number = [](const std::string& s) {
        return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
    };

    size_t slash_position = str.find('/');
    if (slash_position != std::string::npos) {
        std::string num_str = str.substr(0, slash_position);
        std::string denom_str = str.substr(slash_position + 1);
        if (is_number(num_str) && is_number(denom_str)) {
            init(0, std::stoull(num_str), std::stoull(denom_str));
            return;
        }
    } else if (is_number(str)) {
        init(std::stoull(str), 0, 1);
        return;
    }
    throw std::invalid_argument("Invalid Rational format. Expected <numerator>/<denominator> or <integer>");
}

Rational::operator std::string() const
{
    return to_string();
}

std::string Rational::to_string() const
{
    std::string str;

    if (m_integer && m_numerator) {
        str += "{";
    }
    if (m_integer || !m_numerator) {
        str += std::to_string(m_integer);
    }
    if (m_integer && m_numerator) {
        str += " ";
    }
    if (m_numerator) {
        str += std::to_string(m_numerator) + "/" + std::to_string(m_denominator);
    }
    if (m_integer && m_numerator) {
        str += "}";
    }

    return str;
}

std::string Rational::to_total_string() const
{
    uint64_t total_numerator = m_integer * m_denominator + m_numerator;
    if (m_denominator == 1) {
        return std::to_string(total_numerator);
    }
    return std::to_string(total_numerator) + "/" + std::to_string(m_denominator);
}

std::ostream& operator<<(std::ostream& os, const Rational& other)
{
    os << other.to_string();
    return os;
}

std::istream& operator>>(std::istream& is, Rational& other)
{
    std::string str;
    is >> str;
    other = Rational(str);
    return is;
}

void Rational::init(uint64_t integer, uint64_t numerator, uint64_t denominator)
{
    uint64_t quotient;

    if (!denominator) {
        throw RationalException{"Rational: denominator cannot be zero: " + std::to_string(integer) + " " + std::to_string(numerator) + " / 0"};
    }

    m_integer = integer;
    m_numerator = numerator;
    m_denominator = denominator;
    if (m_numerator == 0U) {
        m_denominator = 1U;
    }

    reduce(m_numerator, m_denominator);

    quotient = m_numerator / m_denominator;
    m_integer += quotient;
    quotient *= m_denominator;
    m_numerator -= quotient;
}

void Rational::init()
{
    init(m_integer, m_numerator, m_denominator);
}

/*
 * Calculates Greatest common divisor
 * https://en.wikipedia.org/wiki/Greatest_common_divisor
 */
uint64_t Rational::gcd(uint64_t a, uint64_t b)
{
    if (a < b)
        std::swap(a, b);

    while (b) {
        uint64_t r;

        r = a % b;
        a = b;
        b = r;
    }

    return a;
}

/*
 * Calculates Lowest common denominator
 * https://en.wikipedia.org/wiki/Lowest_common_denominator
 */
uint64_t Rational::lcd(uint64_t d1, uint64_t d2)
{
    uint64_t r = gcd(d1, d2);

    return d1 / r * d2;
}

uint64_t Rational::reduce_two(uint64_t& i1, uint64_t& i2)
{
    uint64_t i = 0;

    while (!((i1 | i2) & 0x1)) {
        i1 >>= 1;
        i2 >>= 1;
        i++;
    }

    return i;
}

void Rational::reduce(uint64_t& i1, uint64_t& i2)
{
    uint64_t r;

    if (!i1)
        return;

    if (!i2) {
        throw RationalException{"Rational: division by zero: " + std::to_string(i1) + " / 0"};
    }

    reduce_two(i1, i2);
    r = gcd(i1, i2);

    i1 /= r;
    i2 /= r;
}

Rational Rational::add_sub(const Rational& lhs, const Rational& rhs, bool is_add) const
{
    uint64_t numerator1 = lhs.m_numerator;
    uint64_t numerator2 = rhs.m_numerator;
    uint64_t integer;
    uint64_t numerator;
    uint64_t denominator;
    uint64_t multiplier;
    Rational result;

    denominator = lcd(lhs.m_denominator, rhs.m_denominator);

    multiplier = denominator / lhs.m_denominator;
    numerator1 *= multiplier;
    multiplier = denominator / rhs.m_denominator;
    numerator2 *= multiplier;
    if (is_add) {
        integer = lhs.m_integer + rhs.m_integer;
        numerator = numerator1 + numerator2;
    } else {
        if (lhs.m_integer < rhs.m_integer) {
            throw RationalException{"Rational: negative rationals are not supported: attempted operation " + lhs.to_string() + " - " + rhs.to_string()};
        }
        integer = lhs.m_integer - rhs.m_integer;

        if (numerator1 < numerator2) {
            if (integer < 1) {
                throw RationalException{"Rational: negative rationals are not supported: attempted operation " + lhs.to_string() + " - " + rhs.to_string()};
            }
            // fractional part of rhs is less than 1 so adding 1 to the minuend
            // makes the difference positive
            --integer;
            numerator1 += denominator;
        }
        numerator = numerator1 - numerator2;
    }

    result.init(integer, numerator, denominator);
    return result;
}

Rational& Rational::add_sub_assign(const Rational& other, bool is_add)
{
    *this = add_sub(*this, other, is_add);
    return *this;
}

Rational Rational::operator+(const Rational& other) const
{
    return add_sub(*this, other, true);
}

Rational& Rational::operator+=(const Rational& other)
{
    return add_sub_assign(other, true);
}

Rational Rational::operator-(const Rational& other) const
{
    return add_sub(*this, other, false);
}

Rational& Rational::operator-=(const Rational& other)
{
    return add_sub_assign(other, false);
}

Rational Rational::mul_div(const Rational& lhs, const Rational& rhs, bool is_multiply) const
{
    uint64_t numerator1;
    uint64_t denominator1;
    uint64_t numerator2;
    uint64_t denominator2;
    uint64_t numerator;
    uint64_t denominator;
    Rational result;

    numerator1 = lhs.m_integer * lhs.m_denominator + lhs.m_numerator;
    denominator1 = lhs.m_denominator;
    numerator2 = rhs.m_integer * rhs.m_denominator + rhs.m_numerator;
    denominator2 = rhs.m_denominator;

    if (is_multiply) {
        reduce(numerator1, denominator2);
        reduce(numerator2, denominator1);

        numerator = numerator1 * numerator2;
        denominator = denominator1 * denominator2;
    } else {
        reduce(numerator1, numerator2);
        reduce(denominator1, denominator2);

        numerator = numerator1 * denominator2;
        denominator = denominator1 * numerator2;
    }

    result.init(0, numerator, denominator);
    return result;
}

Rational& Rational::mul_div_assign(const Rational& lhs, const Rational& rhs, bool is_multiply)
{
    *this = mul_div(lhs, rhs, is_multiply);
    return *this;
}

Rational Rational::operator*(const Rational& other) const
{
    return mul_div(*this, other, true);
}

Rational& Rational::operator*=(const Rational& other)
{
    return mul_div_assign(*this, other, true);
}

Rational Rational::operator/(const Rational& other) const
{
    return mul_div(*this, other, false);
}

Rational& Rational::operator/=(const Rational& other)
{
    return mul_div_assign(*this, other, false);
}

bool Rational::operator==(const Rational& other) const
{
    return m_integer == other.m_integer && m_numerator == other.m_numerator &&
        m_denominator == other.m_denominator;
}

bool Rational::operator!=(const Rational& other) const
{
    return !(*this == other);
}

bool Rational::operator<(const Rational& other) const
{
    if (m_integer == other.m_integer) {
        uint64_t r = lcd(m_denominator, other.m_denominator);

        return (r / m_denominator) * m_numerator <
            (r / other.m_denominator) * other.m_numerator;
    }

    return m_integer < other.m_integer;
}

} // namespace services
} // namespace dev_kit
} // namespace rivermax
