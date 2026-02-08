"""
test_example.py - Python script demonstrating the use of the C++ example module
"""

try:
    import example_module as em

    print("=" * 50)
    print("Testing Pybind11 Example Module")
    print("=" * 50)

    # Test simple functions
    print("\n1. Testing add function:")
    result = em.add(5, 3)
    print(f"   add(5, 3) = {result}")

    print("\n2. Testing multiply function:")
    result = em.multiply(4.5, 2.0)
    print(f"   multiply(4.5, 2.0) = {result}")

    print("\n3. Testing greet function:")
    result = em.greet("World")
    print(f"   greet('World') = '{result}'")

    # Test Calculator class
    print("\n4. Testing Calculator class:")
    calc = em.Calculator()
    print(f"   Initial value: {calc.get_value()}")

    calc.add(10)
    print(f"   After add(10): {calc.get_value()}")

    calc.multiply(2)
    print(f"   After multiply(2): {calc.get_value()}")

    calc.add(5)
    print(f"   After add(5): {calc.get_value()}")

    calc.reset()
    print(f"   After reset(): {calc.get_value()}")

    # Test Calculator with initial value
    print("\n5. Testing Calculator with initial value:")
    calc2 = em.Calculator(100.0)
    print(f"   Initial value: {calc2.get_value()}")
    calc2.multiply(0.5)
    print(f"   After multiply(0.5): {calc2.get_value()}")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

except ImportError as e:
    print("Error: Could not import example_module")
    print(f"Details: {e}")
    print("\nMake sure you have compiled the module first:")
    print("  make build")
except Exception as e:
    print(f"Error: {e}")
