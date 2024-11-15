import numpy as np
import matplotlib.pyplot as plt

# Lorenz 1996 model (1996 version), a chaotic system
def lorenz96(x, F=8.0):
    """Lorenz 1996 model with forcing term F"""
    n = len(x)
    dxdt = np.zeros(n)
    for i in range(n):
        dxdt[i] = (x[(i+1)%n] - x[i-2]) * x[i-1] - x[i] + F
    return dxdt

# Runge-Kutta method for solving the system
def runge_kutta_step(f, x, dt, *params):
    """Single step of the Runge-Kutta method for ODEs"""
    k1 = f(x, *params)
    k2 = f(x + 0.5 * dt * k1, *params)
    k3 = f(x + 0.5 * dt * k2, *params)
    k4 = f(x + dt * k3, *params)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Simulate the Lorenz 1996 system over time
def simulate_lorenz96(n_steps, dt, n_vars, F=8.0):
    """Simulate the Lorenz 1996 model over n_steps"""
    x = np.random.rand(n_vars)  # Initial conditions
    trajectory = np.zeros((n_steps, n_vars))
    for t in range(n_steps):
        trajectory[t] = x
        x = runge_kutta_step(lorenz96, x, dt, F)
    return trajectory

# Simulate noisy sensor measurements
def generate_sensor_data(x_true, noise_std):
    """Simulate noisy sensor measurements from the true states"""
    sensor_data = x_true + np.random.normal(0, noise_std, x_true.shape)
    return sensor_data

# Bayesian weighting function
def fuse_measurements(measurements, noise_stds):
    """Fuse measurements using Bayesian weighting"""
    weights = 1 / np.array(noise_stds) ** 2  # Inverse variance weighting
    weights /= weights.sum()  # Normalize weights
    fused_measurement = np.sum([w * m for w, m in zip(weights, measurements)], axis=0)
    return fused_measurement

# Kalman Filter class
class KalmanFilter:
    def __init__(self, dim_state, process_noise, measurement_noise):
        self.F = np.eye(dim_state)  # State transition model
        self.H = np.eye(dim_state)  # Observation model
        self.Q = process_noise * np.eye(dim_state)  # Process noise covariance
        self.R = measurement_noise * np.eye(dim_state)  # Measurement noise covariance
        self.P = np.eye(dim_state)  # State covariance
        self.x = np.zeros(dim_state)  # Initial state estimate

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P

# Parameters
n_steps = 500
dt = 0.05  # Time step
n_vars = 40  # Number of variables in the Lorenz system
process_noise = 0.1
measurement_noise = 2.0
sensor_noises = [1.0, 3.0]  # Different noise levels for sensors

# Simulate Lorenz 1996 dynamics
true_states = simulate_lorenz96(n_steps, dt, n_vars)

# Generate noisy sensor data for different sensors
sensor1_data = generate_sensor_data(true_states, sensor_noises[0])
sensor2_data = generate_sensor_data(true_states, sensor_noises[1])

# Kalman Filter fusion
kf = KalmanFilter(dim_state=n_vars, process_noise=process_noise, measurement_noise=measurement_noise)
estimates = []
errors_sensor1 = []
errors_sensor2 = []
errors_fusion = []

for t in range(n_steps):
    # Predict
    kf.predict()
    # Fuse data from multiple sensors with Bayesian weighting
    fused_measurement = fuse_measurements([sensor1_data[t], sensor2_data[t]], sensor_noises)
    # Update
    kf.update(fused_measurement)
    estimates.append(kf.x.copy())
    
    # Calculate errors (absolute error)
    error_sensor1 = np.abs(sensor1_data[t] - true_states[t])
    error_sensor2 = np.abs(sensor2_data[t] - true_states[t])
    error_fusion = np.abs(fused_measurement - true_states[t])

    errors_sensor1.append(error_sensor1)
    errors_sensor2.append(error_sensor2)
    errors_fusion.append(error_fusion)

# Convert errors to numpy arrays
errors_sensor1 = np.array(errors_sensor1)
errors_sensor2 = np.array(errors_sensor2)
errors_fusion = np.array(errors_fusion)

# Print average errors for the first few time steps
print(f"Average error (Sensor 1): {np.mean(errors_sensor1[:10], axis=0)}")
print(f"Average error (Sensor 2): {np.mean(errors_sensor2[:10], axis=0)}")
print(f"Average error (Fusion): {np.mean(errors_fusion[:10], axis=0)}")

# Plot 1: Trajectory of the Lorenz system and sensor data
plt.figure(figsize=(12, 6))
plt.plot(true_states[:, 0], true_states[:, 1], label="True Path", color="black", alpha=0.6)
plt.scatter(sensor1_data[:, 0], sensor1_data[:, 1], label="Sensor 1", color="blue", alpha=0.5)
plt.scatter(sensor2_data[:, 0], sensor2_data[:, 1], label="Sensor 2", color="red", alpha=0.5)
plt.plot(np.array(estimates)[:, 0], np.array(estimates)[:, 1], label="Bayesian Fusion Estimate", color="green", linewidth=2)
plt.xlabel("State 1 (X Position)")
plt.ylabel("State 2 (Y Position)")
plt.legend()
plt.title("Improved Bayesian Sensor Fusion with Lorenz 1996 System")
plt.grid()

# Plot 2: Error between measurements and true states
plt.figure(figsize=(12, 6))
plt.plot(np.arange(n_steps), np.mean(errors_sensor1, axis=1), label="Sensor 1 Error", color="blue")
plt.plot(np.arange(n_steps), np.mean(errors_sensor2, axis=1), label="Sensor 2 Error", color="red")
plt.plot(np.arange(n_steps), np.mean(errors_fusion, axis=1), label="Fusion Error", color="green")
plt.xlabel("Time Step")
plt.ylabel("Error (Absolute Difference)")
plt.legend()
plt.title("Error Between Measurements and True States")
plt.grid()

plt.show()
