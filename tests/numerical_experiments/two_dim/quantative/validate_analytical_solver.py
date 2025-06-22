from tests.numerical_experiments.two_dim.quantative.analytical_solver import (
    StefanCornerSolver,
    StefanParameters,
)


# Run validation
test_cases = [
    {"beta": 0.4, "Ti_star": 0.75, "expected_x0": 0.69},
    {"beta": 0.7, "Ti_star": 2.0, "expected_x0": 0.5},
    {"beta": 0.7, "Ti_star": 1.0, "expected_x0": 0.6},
]

print("Running validation cases...")
print("-" * 50)

for i, case in enumerate(test_cases):
    print(f"\nTest Case {i+1}:")
    params = StefanParameters(beta=case["beta"], Ti_star=case["Ti_star"])
    solver = StefanCornerSolver(params)

    try:
        result = solver.solve()
        error = abs(result["x0_star"] - case["expected_x0"]) / case["expected_x0"] * 100
        print(f"Expected x0*: {case['expected_x0']:.3f}")
        print(f"Computed x0*: {result['x0_star']:.3f}")
        print(f"Error: {error:.1f}%")

        # Plot interface for first case
        if i == 0:
            solver.plot_interface()
    except Exception as e:
        print(f"Failed: {e}")

# Example with custom parameters
print("\n" + "=" * 60)
print("Example with custom parameters:")
params = StefanParameters(beta=0.25, Ti_star=0.3)
solver = StefanCornerSolver(params)
result = solver.solve()
solver.plot_interface()
