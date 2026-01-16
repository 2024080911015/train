import sys
import os

# Ensure we can find the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solvers import question_1, question_2, sensitivity

def main():
    print("=== 2022 MCM Problem A Solution Runner ===")
    
    print("\n[1] Running Question 1...")
    question_1.solve_q1()

    print("\n[2] Running Question 2...")
    # Dummy data example
    pass 
    # question_2.run_simulation(...)

    print("\n[3] Running Sensitivity Analysis...")
    sensitivity.analyze_sensitivity()

if __name__ == "__main__":
    main()
