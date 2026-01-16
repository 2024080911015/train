import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import from models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.cyclist import Cyclist
from models.simulator import WPrimeBalanceSimulator

def plot_simulation(time, power, w_bal, cp):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Power (W)', color=color)
    ax1.plot(time, power, color=color, label='Power Output')
    ax1.axhline(y=cp, color='gray', linestyle='--', label='Critical Power (CP)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel("W' Balance (J)", color=color)
    ax2.plot(time, w_bal, color=color, linestyle='-', label="W' Balance")
    ax2.tick_params(axis='y', labelcolor=color)

    # 填充颜色表示 W' 耗尽风险
    ax2.fill_between(time, 0, w_bal, color='blue', alpha=0.1)

    plt.title("Race Simulation: Power Output vs W' Balance")
    fig.tight_layout()
    plt.show()

def run_simulation(cyclist_name, cp, w_prime, power_data):
    # Setup cyclist
    rider = Cyclist(cyclist_name, cp=cp, w_prime=w_prime)
    sim = WPrimeBalanceSimulator(rider)

    # Run simulation
    w_balance = sim.simulate_race(power_data)

    # Plot
    time_series = np.arange(len(power_data))
    plot_simulation(time_series, power_data, w_balance, rider.cp)

if __name__ == "__main__":
    # Example usage
    # power_data = ... load from data file ...
    print("Running Question 2 Simulation...")
    # Dummy data for test
    run_simulation("Test Rider", 300, 20000, np.array([350]*100 + [200]*100))
