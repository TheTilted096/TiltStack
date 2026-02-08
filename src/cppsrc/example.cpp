// example.cpp - Simple C++ functions to demonstrate pybind11

#include <string>
#include <vector>

// Simple function that adds two numbers
int add(int a, int b) {
    return a + b;
}

// Function that multiplies two numbers
double multiply(double a, double b) {
    return a * b;
}

// Function that returns a greeting message
std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

// A simple class to demonstrate class bindings
class Calculator {
private:
    double value;

public:
    Calculator() : value(0.0) {}

    Calculator(double initial_value) : value(initial_value) {}

    void add(double x) {
        value += x;
    }

    void multiply(double x) {
        value *= x;
    }

    double get_value() const {
        return value;
    }

    void reset() {
        value = 0.0;
    }
};
