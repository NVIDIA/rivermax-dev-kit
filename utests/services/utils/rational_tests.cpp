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

#include <gtest/gtest.h>

#include "rdk/services/utils/rational.h"

using rdk::services::Rational;
using rdk::services::RationalException;
using rdk::services::rational_cast;

/* Tests default and parameterized constructors for the Rational class. */
TEST(RationalTests, Init)
{
    // Default constructor
    Rational a;
    ASSERT_EQ(a.integer(), 0);
    ASSERT_EQ(a.numerator(), 0);

    // Constructor with integer, numerator, and denominator
    Rational b(2, 3, 4);
    ASSERT_EQ(b.integer(), 2);
    ASSERT_EQ(b.numerator(), 3);
    ASSERT_EQ(b.denominator(), 4);

    // Constructor with integer only
    Rational c(5);
    ASSERT_EQ(c.integer(), 5);
    ASSERT_EQ(c.numerator(), 0);

    // Constructor with numerator and denominator (auto-reduces)
    Rational d(12, 34);
    ASSERT_EQ(d.integer(), 0);
    ASSERT_EQ(d.numerator(), 6);
    ASSERT_EQ(d.denominator(), 17);
}

/* Test constructors from different integral types */
TEST(RationalTests, CreateFromIntegral)
{
    Rational u8(uint8_t(1));
    ASSERT_EQ(u8.integer(), 1);
    ASSERT_EQ(u8.numerator(), 0);
    ASSERT_EQ(u8.denominator(), 1);

    Rational u16(uint16_t(1));
    ASSERT_EQ(u16.integer(), 1);
    ASSERT_EQ(u16.numerator(), 0);
    ASSERT_EQ(u16.denominator(), 1);

    Rational u32(uint32_t(1));
    ASSERT_EQ(u32.integer(), 1);
    ASSERT_EQ(u32.numerator(), 0);
    ASSERT_EQ(u32.denominator(), 1);

    Rational u64(uint64_t(1));
    ASSERT_EQ(u64.integer(), 1);
    ASSERT_EQ(u64.numerator(), 0);
    ASSERT_EQ(u64.denominator(), 1);

    Rational i8(int8_t(1));
    ASSERT_EQ(i8.integer(), 1);
    ASSERT_EQ(i8.numerator(), 0);
    ASSERT_EQ(i8.denominator(), 1);

    Rational i16(int16_t(1));
    ASSERT_EQ(i16.integer(), 1);
    ASSERT_EQ(i16.numerator(), 0);
    ASSERT_EQ(i16.denominator(), 1);

    Rational i32(int32_t(1));
    ASSERT_EQ(i32.integer(), 1);
    ASSERT_EQ(i32.numerator(), 0);
    ASSERT_EQ(i32.denominator(), 1);

    Rational i64(int64_t(1));
    ASSERT_EQ(i64.integer(), 1);
    ASSERT_EQ(i64.numerator(), 0);
    ASSERT_EQ(i64.denominator(), 1);

    EXPECT_THROW(Rational(int8_t(-1)), RationalException);
    EXPECT_THROW(Rational(int16_t(-1)), RationalException);
    EXPECT_THROW(Rational(int32_t(-1)), RationalException);
    EXPECT_THROW(Rational(int64_t(-1)), RationalException);
}

/* Test casts to different integral types */
TEST(RationalTests, CastToIntegral)
{
    Rational a(1, 2, 3);

    EXPECT_EQ(static_cast<uint8_t>(a), uint8_t(1));
    EXPECT_EQ(static_cast<uint16_t>(a), uint16_t(1));
    EXPECT_EQ(static_cast<uint32_t>(a), uint32_t(1));
    EXPECT_EQ(static_cast<uint64_t>(a), uint64_t(1));
    EXPECT_EQ(static_cast<int8_t>(a), int8_t(1));
    EXPECT_EQ(static_cast<int16_t>(a), int16_t(1));
    EXPECT_EQ(static_cast<int32_t>(a), int32_t(1));
    EXPECT_EQ(static_cast<int64_t>(a), int64_t(1));
}

/* Tests addition operations between Rational numbers and integers. */
TEST(RationalTests, Add)
{
    // Rational + Rational
    Rational a(1, 2);
    Rational b(3, 4);
    Rational c = a + b;

    ASSERT_EQ(c.integer(), 1);
    ASSERT_EQ(c.numerator(), 1);
    ASSERT_EQ(c.denominator(), 4);

    // Rational += integer
    c += 5;

    ASSERT_EQ(c.integer(), 6);
    ASSERT_EQ(c.numerator(), 1);
    ASSERT_EQ(c.denominator(), 4);

    // integer + Rational
    Rational d = 6 + c;

    ASSERT_EQ(d.integer(), 12);
    ASSERT_EQ(d.numerator(), 1);
    ASSERT_EQ(d.denominator(), 4);

    // Rational + integer
    Rational e = d + 7;

    ASSERT_EQ(e.integer(), 19);
    ASSERT_EQ(e.numerator(), 1);
    ASSERT_EQ(e.denominator(), 4);

    // Addition with different denominators (LCM calculation)
    Rational f = Rational(1, 6) + Rational(1, 15);

    ASSERT_EQ(f.integer(), 0);
    ASSERT_EQ(f.numerator(), 7);
    ASSERT_EQ(f.denominator(), 30);
}

/**
 * @brief: Tests subtraction operations between Rational numbers and integers.
 */
TEST(RationalTests, Sub)
{
    // Rational - Rational
    Rational a(7, 4);
    Rational b(1, 2);
    Rational c = a - b;

    ASSERT_EQ(c.integer(), 1);
    ASSERT_EQ(c.numerator(), 1);
    ASSERT_EQ(c.denominator(), 4);

    // Rational -= integer
    c -= 1;

    ASSERT_EQ(c.integer(), 0);
    ASSERT_EQ(c.numerator(), 1);
    ASSERT_EQ(c.denominator(), 4);

    // integer - Rational
    Rational d = 5 - c;

    ASSERT_EQ(d.integer(), 4);
    ASSERT_EQ(d.numerator(), 3);
    ASSERT_EQ(d.denominator(), 4);

    // Rational - integer
    Rational e = d - 2;

    ASSERT_EQ(e.integer(), 2);
    ASSERT_EQ(e.numerator(), 3);
    ASSERT_EQ(e.denominator(), 4);

    // Subtraction with mixed numbers (same denominator)
    Rational f = Rational(2, 2, 3) - Rational(1, 1, 3);

    ASSERT_EQ(f.integer(), 1);
    ASSERT_EQ(f.numerator(), 1);
    ASSERT_EQ(f.denominator(), 3);

    // Subtraction requiring borrow from integer part
    Rational g = Rational(2, 1, 3) - Rational(1, 2, 3);

    ASSERT_EQ(g.integer(), 0);
    ASSERT_EQ(g.numerator(), 2);
    ASSERT_EQ(g.denominator(), 3);
}

/* Tests multiplication operations between Rational numbers and integers. */
TEST(RationalTests, Mul)
{
    // Rational * Rational
    Rational a(2, 3);
    Rational b(3, 4);
    Rational c = a * b;

    ASSERT_EQ(c.integer(), 0);
    ASSERT_EQ(c.numerator(), 1);
    ASSERT_EQ(c.denominator(), 2);

    // Rational *= integer
    c *= 5;

    ASSERT_EQ(c.integer(), 2);
    ASSERT_EQ(c.numerator(), 1);
    ASSERT_EQ(c.denominator(), 2);

    // Rational * integer
    Rational d = c * 3;

    ASSERT_EQ(d.integer(), 7);
    ASSERT_EQ(d.numerator(), 1);
    ASSERT_EQ(d.denominator(), 2);

    // integer * Rational
    Rational e = 3 * d;

    ASSERT_EQ(e.integer(), 22);
    ASSERT_EQ(e.numerator(), 1);
    ASSERT_EQ(e.denominator(), 2);
}

/* Tests division operations between Rational numbers and integers. */
TEST(RationalTests, Div)
{
    // Rational /= integer
    Rational a(1);
    a /= 2;
    ASSERT_EQ(a.integer(), 0);
    ASSERT_EQ(a.numerator(), 1);
    ASSERT_EQ(a.denominator(), 2);

    // Rational / integer
    Rational b = a / 2;
    ASSERT_EQ(b.integer(), 0);
    ASSERT_EQ(b.numerator(), 1);
    ASSERT_EQ(b.denominator(), 4);

    // integer / Rational (reciprocal)
    Rational c(30000, 1001);
    Rational d = 1 / c;
    ASSERT_EQ(d.integer(), 0);
    ASSERT_EQ(d.numerator(), 1001);
    ASSERT_EQ(d.denominator(), 30000);
}

/* Tests type casting of Rational to int, double, and bool. */
TEST(RationalTests, Cast)
{
    // Cast to int (truncates)
    Rational a(1, 3);
    auto i = rational_cast<int>(a);
    ASSERT_EQ(i, 0);

    // Cast to double
    auto d = rational_cast<double>(a);
    EXPECT_NEAR(d, 0.33, 0.01);

    // Cast to bool (non-zero fraction is true)
    ASSERT_TRUE(static_cast<bool>(Rational(0, 1, 2)));

    // Cast to bool (non-zero integer is true)
    ASSERT_TRUE(static_cast<bool>(Rational(3)));

    // Cast to bool (zero is false)
    ASSERT_FALSE(static_cast<bool>(Rational(0)));
    ASSERT_TRUE(static_cast<bool>(Rational(0, 1, 2)));
    ASSERT_TRUE(static_cast<bool>(Rational(1)));
    ASSERT_TRUE(Rational(1U));
    ASSERT_TRUE(Rational(0, 1, 2));
    ASSERT_TRUE(Rational(1, 1, 2));
    ASSERT_FALSE(Rational(0U));
    ASSERT_FALSE(Rational(0, 0, 1));
    ASSERT_FALSE(Rational(0, 0, 100));
}

/* Tests the less-than comparison operator for Rational numbers. */
TEST(RationalTests, CompareLess)
{
    // Rational < Rational (equal values)
    ASSERT_FALSE(Rational(1) < Rational(1));
    ASSERT_FALSE(Rational(1, 2) < Rational(1, 2));
    ASSERT_FALSE(Rational(1, 2, 3) < Rational(1, 2, 3));

    // Rational < Rational (different values)
    ASSERT_TRUE(Rational(1, 2) < Rational(2, 3));
    ASSERT_FALSE(Rational(2, 3) < Rational(1, 2));
    ASSERT_TRUE(Rational(0, 1, 2) < Rational(1, 1, 2));
    ASSERT_FALSE(Rational(1, 1, 2) < Rational(0, 1, 2));

    // integer < Rational
    ASSERT_FALSE(1 < Rational(1));
    ASSERT_TRUE(1 < Rational(1, 2, 3));
    ASSERT_TRUE(1 < Rational(2));
    ASSERT_FALSE(3 < Rational(2));

    // Rational < integer
    ASSERT_FALSE(Rational(1) < 1);
    ASSERT_FALSE(Rational(1, 2, 3) < 1);
    ASSERT_TRUE(Rational(1) < 2);
    ASSERT_FALSE(Rational(2) < 1);
}

/* Tests the less-than-or-equal comparison operator for Rational numbers. */
TEST(RationalTests, CompareLessEqual)
{
    // Rational <= Rational (equal values)
    ASSERT_TRUE(Rational(1) <= Rational(1));
    ASSERT_TRUE(Rational(1, 2) <= Rational(1, 2));
    ASSERT_TRUE(Rational(1, 2, 3) <= Rational(1, 2, 3));

    // Rational <= Rational (different values)
    ASSERT_TRUE(Rational(1, 2) <= Rational(2, 3));
    ASSERT_FALSE(Rational(2, 3) <= Rational(1, 2));
    ASSERT_TRUE(Rational(0, 1, 2) <= Rational(1, 1, 2));
    ASSERT_FALSE(Rational(1, 1, 2) <= Rational(0, 1, 2));

    // integer <= Rational
    ASSERT_TRUE(1 <= Rational(1));
    ASSERT_TRUE(1 <= Rational(1, 2, 3));
    ASSERT_TRUE(1 <= Rational(2));
    ASSERT_FALSE(3 <= Rational(2));

    // Rational <= integer
    ASSERT_TRUE(Rational(1) <= 1);
    ASSERT_FALSE(Rational(1, 2, 3) <= 1);
    ASSERT_TRUE(Rational(1) <= 2);
    ASSERT_FALSE(Rational(2) <= 1);
}

/* Tests the greater-than comparison operator for Rational numbers. */
TEST(RationalTests, CompareGreater)
{
    // Rational > Rational (equal values)
    ASSERT_FALSE(Rational(1) > Rational(1));
    ASSERT_FALSE(Rational(1, 2) > Rational(1, 2));
    ASSERT_FALSE(Rational(1, 2, 3) > Rational(1, 2, 3));

    // Rational > Rational (different values)
    ASSERT_FALSE(Rational(1, 2) > Rational(2, 3));
    ASSERT_TRUE(Rational(2, 3) > Rational(1, 2));
    ASSERT_FALSE(Rational(0, 1, 2) > Rational(1, 1, 2));
    ASSERT_TRUE(Rational(1, 1, 2) > Rational(0, 1, 2));

    // integer > Rational
    ASSERT_FALSE(1 > Rational(1));
    ASSERT_FALSE(1 > Rational(1, 2, 3));
    ASSERT_FALSE(1 > Rational(2));
    ASSERT_TRUE(3 > Rational(2));

    // Rational > integer
    ASSERT_FALSE(Rational(1) > 1);
    ASSERT_TRUE(Rational(1, 2, 3) > 1);
    ASSERT_FALSE(Rational(1) > 2);
    ASSERT_TRUE(Rational(2) > 1);
}

/* Tests the greater-than-or-equal comparison operator for Rational numbers. */
TEST(RationalTests, CompareGreaterEqual)
{
    // Rational >= Rational (equal values)
    ASSERT_TRUE(Rational(1) >= Rational(1));
    ASSERT_TRUE(Rational(1, 2) >= Rational(1, 2));
    ASSERT_TRUE(Rational(1, 2, 3) >= Rational(1, 2, 3));

    // Rational >= Rational (different values)
    ASSERT_FALSE(Rational(1, 2) >= Rational(2, 3));
    ASSERT_TRUE(Rational(2, 3) >= Rational(1, 2));
    ASSERT_FALSE(Rational(0, 1, 2) >= Rational(1, 1, 2));
    ASSERT_TRUE(Rational(1, 1, 2) >= Rational(0, 1, 2));

    // integer >= Rational
    ASSERT_TRUE(1 >= Rational(1));
    ASSERT_FALSE(1 >= Rational(1, 2, 3));
    ASSERT_FALSE(1 >= Rational(2));
    ASSERT_TRUE(3 >= Rational(2));

    // Rational >= integer
    ASSERT_TRUE(Rational(1) >= 1);
    ASSERT_TRUE(Rational(1, 2, 3) >= 1);
    ASSERT_FALSE(Rational(1) >= 2);
    ASSERT_TRUE(Rational(2) >= 1);
}

/* Tests the equality comparison operator for Rational numbers. */
TEST(RationalTests, CompareEqual)
{
    // Rational == Rational (equal values)
    ASSERT_TRUE(Rational(1) == Rational(1));
    ASSERT_TRUE(Rational(1, 2) == Rational(1, 2));
    ASSERT_TRUE(Rational(1, 2, 3) == Rational(1, 2, 3));

    // Rational == Rational (different values)
    ASSERT_FALSE(Rational(1, 2) == Rational(2, 3));
    ASSERT_FALSE(Rational(2, 3) == Rational(1, 2));
    ASSERT_FALSE(Rational(1) == Rational(1, 2, 3));
    ASSERT_FALSE(Rational(0, 1, 2) == Rational(1, 1, 2));
    ASSERT_FALSE(Rational(1, 1, 2) == Rational(0, 1, 2));
    // Rational == Rational (zero numerator, different denominator)
    ASSERT_TRUE(Rational(0, 0, 1) == Rational(0, 0, 100));
    ASSERT_TRUE(Rational(0, 0, 1) == Rational(0, 0, 100));
    ASSERT_TRUE(Rational(0, 1, 10) - Rational(0, 1, 10) == Rational(0, 0, 100));

    // integer == Rational
    ASSERT_TRUE(1 == Rational(1));
    ASSERT_TRUE(0 == Rational(0, 0, 100));
    ASSERT_FALSE(1 == Rational(1, 2, 3));
    ASSERT_FALSE(1 == Rational(2));
    ASSERT_FALSE(3 == Rational(2));

    // Rational == integer
    ASSERT_TRUE(Rational(1) == 1);
    ASSERT_FALSE(Rational(1, 2, 3) == 1);
    ASSERT_FALSE(Rational(1) == 2);
    ASSERT_FALSE(Rational(2) == 1);
}

/* Tests the inequality comparison operator for Rational numbers. */
TEST(RationalTests, CompareNotEqual)
{
    // Rational != Rational (equal values)
    ASSERT_FALSE(Rational(1) != Rational(1));
    ASSERT_FALSE(Rational(1, 2) != Rational(1, 2));
    ASSERT_FALSE(Rational(1, 2, 3) != Rational(1, 2, 3));

    // Rational != Rational (different values)
    ASSERT_TRUE(Rational(1, 2) != Rational(2, 3));
    ASSERT_TRUE(Rational(2, 3) != Rational(1, 2));
    ASSERT_TRUE(Rational(1) != Rational(1, 2, 3));
    ASSERT_TRUE(Rational(0, 1, 2) != Rational(1, 1, 2));
    ASSERT_TRUE(Rational(1, 1, 2) != Rational(0, 1, 2));

    // integer != Rational
    ASSERT_FALSE(1 != Rational(1));
    ASSERT_TRUE(1 != Rational(1, 2, 3));
    ASSERT_TRUE(1 != Rational(2));
    ASSERT_TRUE(3 != Rational(2));

    // Rational != integer
    ASSERT_FALSE(Rational(1) != 1);
    ASSERT_TRUE(Rational(1, 2, 3) != 1);
    ASSERT_TRUE(Rational(1) != 2);
    ASSERT_TRUE(Rational(2) != 1);
}

/* Tests for prefix and suffix increment/decrement operators */
TEST(RationalTests, IncrementDecrement)
{
    Rational a(5U);
    ASSERT_EQ(a++, 5U);
    ASSERT_EQ(a, 6U);
    ASSERT_EQ(++a, 7U);
    ASSERT_EQ(a, 7U);

    ASSERT_EQ(a--, 7U);
    ASSERT_EQ(a, 6U);
    ASSERT_EQ(--a, 5U);
    ASSERT_EQ(a, 5U);

    Rational b(5, 1, 2);
    ASSERT_EQ(b++, Rational(5, 1, 2));
    ASSERT_EQ(b, Rational(6, 1, 2));
    ASSERT_EQ(++b, Rational(7, 1, 2));
    ASSERT_EQ(b, Rational(7, 1, 2));

    ASSERT_EQ(b--, Rational(7, 1, 2));
    ASSERT_EQ(b, Rational(6, 1, 2));
    ASSERT_EQ(--b, Rational(5, 1, 2));
    ASSERT_EQ(b, Rational(5, 1, 2));
}

/* Tests construction of Rational from string representation. */
TEST(RationalTests, StringConstruction)
{
    // Integer format
    Rational a("60");
    ASSERT_EQ(a.integer(), 60);
    ASSERT_EQ(a.numerator(), 0);
    ASSERT_EQ(a.denominator(), 1);

    // Fraction format
    Rational b("60000/1001");
    ASSERT_EQ(b.integer(), 59);
    ASSERT_EQ(b.numerator(), 941);
    ASSERT_EQ(b.denominator(), 1001);

    // Simple fraction
    Rational c("1/2");
    ASSERT_EQ(c.integer(), 0);
    ASSERT_EQ(c.numerator(), 1);
    ASSERT_EQ(c.denominator(), 2);

    // Fraction that reduces to integer
    Rational d("100/1");
    ASSERT_EQ(d.integer(), 100);
    ASSERT_EQ(d.numerator(), 0);
    ASSERT_EQ(d.denominator(), 1);
}

/* Tests that invalid string formats throw std::invalid_argument. */
TEST(RationalTests, StringConstructionInvalid)
{
    ASSERT_THROW(Rational("abc"), std::invalid_argument);
    ASSERT_THROW(Rational("60/abc"), std::invalid_argument);
    ASSERT_THROW(Rational("abc/60"), std::invalid_argument);
    ASSERT_THROW(Rational("60.5"), std::invalid_argument);
    ASSERT_THROW(Rational(""), std::invalid_argument);
    ASSERT_THROW(Rational("/60"), std::invalid_argument);
    ASSERT_THROW(Rational("60/"), std::invalid_argument);
    ASSERT_THROW(Rational("-123"), std::invalid_argument);
    ASSERT_THROW(Rational("-123/45"), std::invalid_argument);
    ASSERT_THROW(Rational("123/-45"), std::invalid_argument);
}

/* Tests that overflow values throw std::out_of_range. */
TEST(RationalTests, StringConstructionOverflow)
{
    ASSERT_THROW(Rational("999999999999999999999999999999999999999999999"), std::out_of_range);
    ASSERT_THROW(Rational("999999999999999999999999999999999999999999999/1"), std::out_of_range);
    ASSERT_THROW(Rational("1/999999999999999999999999999999999999999999999"), std::out_of_range);
}

/* Tests that division by zero throws RationalException. */
TEST(RationalTests, DivisionByZero)
{
    using rdk::services::RationalException;

    // Constructor with zero denominator (numerator, denominator)
    ASSERT_THROW(Rational(1, 0), RationalException);

    // Constructor with zero denominator (integer, numerator, denominator)
    ASSERT_THROW(Rational(1, 2, 0), RationalException);

    // String constructor with zero denominator
    ASSERT_THROW(Rational("1/0"), RationalException);
    ASSERT_THROW(Rational("0/0"), RationalException);

    // Division operator by zero Rational
    Rational a(1);
    ASSERT_THROW(a / Rational(0), RationalException);

    // Division operator by zero integer
    ASSERT_THROW(a / 0, RationalException);

    // Division assignment by zero Rational
    ASSERT_THROW(a /= Rational(0), RationalException);

    // Division assignment by zero integer
    ASSERT_THROW(a /= 0, RationalException);

    // Integer divided by zero Rational
    ASSERT_THROW(1 / Rational(0), RationalException);
}

/* Tests conversion of Rational to string representation. */
TEST(RationalTests, ToString)
{
    // Integer (denominator = 1)
    Rational a(60);
    ASSERT_EQ(a.to_string(), "60");
    ASSERT_EQ(a.to_total_string(), "60");
    ASSERT_EQ(std::string(a), "60");

    // Fraction
    Rational b(60000, 1001);
    ASSERT_EQ(b.to_string(), "{59 941/1001}");
    ASSERT_EQ(b.to_total_string(), "60000/1001");
    ASSERT_EQ(std::string(b), "{59 941/1001}");

    // Simple fraction
    Rational c(1, 2);
    ASSERT_EQ(c.to_string(), "1/2");
    ASSERT_EQ(c.to_total_string(), "1/2");
    ASSERT_EQ(std::string(c), "1/2");

    // Mixed number outputs
    Rational d(1, 1, 2);  // 1 + 1/2 = 3/2
    ASSERT_EQ(d.to_string(), "{1 1/2}");
    ASSERT_EQ(d.to_total_string(), "3/2");
    ASSERT_EQ(std::string(d), "{1 1/2}");
}

/* Tests that string parsing and conversion are consistent (round-trip). */
TEST(RationalTests, StringRoundTrip)
{
    // Test that parsing a string and converting back produces the same string
    std::string original1 = "60000/1001";
    Rational r1(original1);
    ASSERT_EQ(r1.to_total_string(), original1);

    std::string original2 = "60";
    Rational r2(original2);
    ASSERT_EQ(r2.to_total_string(), original2);

    // Edge case: large values
    std::string original3 = "1000000/999999";
    Rational r3(original3);
    ASSERT_EQ(r3.to_total_string(), original3);
}

/* Tests input stream operator for parsing Rational from streams. */
TEST(RationalTests, StreamInput)
{
    // Fraction format
    std::istringstream iss("60000/1001");
    Rational r;
    iss >> r;
    ASSERT_EQ(r.integer(), 59);
    ASSERT_EQ(r.numerator(), 941);
    ASSERT_EQ(r.denominator(), 1001);

    // Integer format
    std::istringstream iss2("42");
    Rational r2;
    iss2 >> r2;
    ASSERT_EQ(r2.integer(), 42);
    ASSERT_EQ(r2.numerator(), 0);
}

/* Tests total_numerator() and total_denominator() for improper fraction form. */
TEST(RationalTests, TotalNumeratorDenominator)
{
    // Integer only (no fraction part)
    Rational a(5);
    ASSERT_EQ(a.total_numerator(), 5);
    ASSERT_EQ(a.total_denominator(), 1);

    // Proper fraction only (no integer part)
    Rational b(3, 4);
    ASSERT_EQ(b.total_numerator(), 3);
    ASSERT_EQ(b.total_denominator(), 4);

    // Mixed number: 2 + 3/4 = 11/4
    Rational c(2, 3, 4);
    ASSERT_EQ(c.total_numerator(), 11);
    ASSERT_EQ(c.total_denominator(), 4);

    // Video frame rate: 60000/1001 = 59 + 941/1001
    Rational d("60000/1001");
    ASSERT_EQ(d.total_numerator(), 60000);
    ASSERT_EQ(d.total_denominator(), 1001);

    // Default constructed
    Rational e;
    ASSERT_EQ(e.total_numerator(), 0);
    ASSERT_EQ(e.total_denominator(), 1);

    // Large mixed number: 100 + 1/2 = 201/2
    Rational f(100, 1, 2);
    ASSERT_EQ(f.total_numerator(), 201);
    ASSERT_EQ(f.total_denominator(), 2);

    // Verify consistency with rational_cast
    Rational g(30000, 1001);
    double expected = static_cast<double>(g.total_numerator()) / g.total_denominator();
    EXPECT_NEAR(rational_cast<double>(g), expected, 0.0001);
}
