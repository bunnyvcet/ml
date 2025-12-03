import random
import math
def rand_uniform(n):
    return [random.random() for _ in range(n)]
def f(x):
    return x**2
random.seed(0)
n = 10
samples = rand_uniform(n)
f_values = [f(x) for x in samples]
I_estimate = sum(f_values) / n
I_exact = 1 / 3
print("Sample\t x\t\t f(x)=x^2")
for i in range(n):
    print(f"{i+1}\t {samples[i]:.6f}\t {f_values[i]:.6f}")
print("\nMonte Carlo estimate of I = ∫₀ ¹ x² dx")
print(f"Estimated value : {I_estimate:.6f}")
print(f"Exact value : {I_exact:.6f}")
print(f"Absolute error : {abs(I_estimate - I_exact):.6f}")
