import torch

A = torch.randn(3, 3)

print(A)

jac1 = torch.linalg.det(A)

slog = torch.linalg.slogdet(A)

sign = slog[0]
amplitude = slog[1]

jac2 = sign * torch.exp(amplitude)

print("Determinant via torch.linalg.det:", jac1)
print("Determinant via slogdet:", jac2)
print("Difference:", jac1 - jac2)
