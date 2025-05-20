from improved_model import run_cost_effectiveness_model
import pandas as pd
import numpy as np

# Run the model
results = run_cost_effectiveness_model()

# Display top-level summary
for scenario, data in results.items():
    print(f"\n--- Scenario: {scenario} ---")
    for strat, vals in data["Results"].items():
        print(f"\nStrategy: {strat}")
        print(f"  Total QALYs: {vals['QALYs']:.2f}")
        print(f"  Total Costs: ${vals['Costs']:.2f}")

    # ICERs
    print("\nICERs (vs Stress Test):")
    icers = data["ICERs"]
    if not icers:
        print("  No ICERs calculated.")
    else:
        for strat, icer in icers.items():
            if isinstance(icer, (float, int, np.floating)):
                print(f"  {strat}: ${float(icer):,.2f} per QALY")
            else:
                print(f"  {strat}: {icer}")

# Function to sanitize filenames
def generate_safe_filename(scenario: str, strat: str) -> str:
    replacements = {
        "<": "lt", ">": "gt", ":": "", '"': "",
        "/": "_", "\\": "_", "|": "_", "?": "",
        "*": "", " ": "_"
    }
    for old, new in replacements.items():
        scenario = scenario.replace(old, new)
        strat = strat.replace(old, new)
    return f"{scenario}_{strat}_Annual.csv"

# Save annual results to CSV
for scenario, data in results.items():
    for strat, vals in data["Results"].items():
        annual_df = pd.DataFrame(vals["Annual"])
        filename = generate_safe_filename(scenario, strat)
        annual_df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
