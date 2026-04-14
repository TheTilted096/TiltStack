#include "CFRUtils.h"
#include <gtest/gtest.h>

// Same seed must produce the same sequence.
TEST(RNG, Determinism) {
    RNG a(12345), b(12345);
    for (int i = 0; i < 1000; i++)
        EXPECT_EQ(a.next(), b.next());
}

// Different seeds must diverge immediately.
TEST(RNG, SeedIndependence) {
    RNG a(1), b(2);
    EXPECT_NE(a.next(), b.next());
}

// nextFloat must always be in [0, 1).
TEST(RNG, FloatRange) {
    RNG r(42);
    for (int i = 0; i < 1000000; i++) {
        float f = r.nextFloat();
        EXPECT_GE(f, 0.0f);
        EXPECT_LT(f, 1.0f);
    }
}

// seed(0) causes the state to become permanently zero — document this trap.
TEST(RNG, ZeroSeedTrap) {
    RNG r(0);
    for (int i = 0; i < 10; i++)
        EXPECT_EQ(r.next(), 0ULL);
}
