"""
improved_model_commented.py

Cost-effectiveness model for coronary artery disease (CAD) diagnosis and treatment pathways.
This model includes:
  - A decision tree to allocate patients based on diagnostic strategy (FFRCT vs Stress Test)
  - A Markov model to simulate long-term outcomes over a specified time horizon
  - Cost and QALY calculations with discounting

Improvements:
  - Added detailed comments for clarity
  - Normalized patient allocations to ensure probabilities sum to 1
  - Removed unused imports (pandas, matplotlib)
"""

import numpy as np
from scipy.stats import beta, gamma

# --------------------------------------------------------------------------------
# Model parameters
# --------------------------------------------------------------------------------
class ModelParams:
    """
    Container for all model parameters: time horizon, discount rate, health states,
    utilities (QALYs), costs, diagnostic test characteristics, and transition rates.
    """
    def __init__(self):
        # Simulation settings
        self.time_horizon = 25      # Number of yearly cycles
        self.discount_rate = 0.03   # Annual discount rate for costs and QALYs
        self.starting_age = 60      # Average age of cohort at baseline
        self.cycle_length = 1       # Length of each cycle (years)

        # Define all possible health states in the Markov model
        self.states = [
            "No CAD",          # No disease
            "CAD Undiagnosed", # Disease present but not yet detected
            "CAD Untreated",   # Diagnosed but managed medically
            "PCI Treated",     # Treated with percutaneous coronary intervention
            "CABG Treated",    # Treated with bypass surgery
            "Post-MI",         # History of myocardial infarction
            "Dead"             # Absorbing state
        ]

        # Quality-of-life utilities for each state
        self.utilities = {
            "No CAD": 0.91,
            "CAD Undiagnosed": 0.75,
            "CAD Untreated": 0.78,
            "PCI Treated": 0.85,
            "CABG Treated": 0.83,
            "Post-MI": 0.65,
            "Dead": 0.0
        }

        # Additional disutility for rural patients based on travel distance
        self.rural_disutilities = {
            "Distance < 50 miles": 0.01,
            "Distance 50-100 miles": 0.02,
            "Distance > 100 miles": 0.04
        }

        # Cost inputs for diagnostics, interventions, and follow-up
        self.costs = {
            # Diagnostic test costs
            "FFRCT": 1450,
            "Stress Test": 350,
            "ICA": 5000,

            # Intervention costs
            "PCI": 17000,
            "CABG": 35000,
            "MI Hospitalization": 20000,

            # Annual follow-up and medical therapy costs
            "CAD Medication": 700,
            "CAD Monitoring": 300,
            "PCI Follow-up": 1200,
            "CABG Follow-up": 1500,
            "Post-MI Care": 3000,

            # Travel and emergency transport for rural patients
            "Travel Per Visit": {
                "Distance < 50 miles": 100,
                "Distance 50-100 miles": 200,
                "Distance > 100 miles": 400
            },
            "Emergency Transport": {
                "Distance < 50 miles": 1000,
                "Distance 50-100 miles": 2500,
                "Distance > 100 miles": 5000
            }
        }

        # Diagnostic test performance characteristics
        self.test_characteristics = {
            "FFRCT": {
                "Sensitivity": 0.90,
                "Specificity": 0.79,
                "Inconclusive Rate": 0.05,
                "Rural Feasibility": 0.95
            },
            "Stress Test": {
                "Sensitivity": 0.70,
                "Specificity": 0.75,
                "Inconclusive Rate": 0.10,
                "Rural Feasibility": 0.90
            }
        }

        # Distribution of rural distances in the cohort
        self.rural_distance_distribution = {
            "Distance < 50 miles": 0.30,
            "Distance 50-100 miles": 0.45,
            "Distance > 100 miles": 0.25
        }

        # Treatment effects: reduction in MI risk and mortality
        self.treatment_effects = {
            "PCI":      {"Reduction in MI Risk": 0.65, "Reduction in Mortality": 0.40},
            "CABG":     {"Reduction in MI Risk": 0.75, "Reduction in Mortality": 0.50},
            "Medical Therapy": {"Reduction in MI Risk": 0.40, "Reduction in Mortality": 0.25}
        }

        # Pre-test disease prevalence
        self.pretest_probability = 0.40

        # Base transition rates per year
        self.base_transitions = {
            "CAD Progression Rate": 0.05,
            "MI Risk Untreated": 0.03,
            "MI Risk Treated": 0.01,
            "Post-MI Mortality": 0.08
        }

        # Age-specific all-cause mortality rates for interpolation
        self.age_mortality = {50: 0.004, 55: 0.006, 60: 0.009, 65: 0.014,
                              70: 0.021, 75: 0.033, 80: 0.054, 85: 0.095}

        # Repeat intervention rates each year
        self.repeat_intervention = {"PCI": 0.12, "CABG": 0.02}

        # Number of healthcare visits per year by state
        self.visits_per_year = {
            "No CAD": 0.5,
            "CAD Untreated": 2.0,
            "PCI Treated": 3.0,
            "CABG Treated": 4.0,
            "Post-MI": 6.0
        }

# --------------------------------------------------------------------------------
# Decision tree for initial diagnostic and treatment allocation
# --------------------------------------------------------------------------------
class DiagnosticPathway:
    def __init__(self, params, rural_distance="Distance 50-100 miles"):
        self.params = params
        self.rural_distance = rural_distance

    def classify_patients(self, strategy):
        """
        Allocate a hypothetical cohort of size 1 across health states based on test sensitivity,
        specificity, and prevalence. Returns dictionary of proportions summing to 1.
        """
        test = self.params.test_characteristics[strategy]
        prevalence = self.params.pretest_probability

        # Calculate initial true/false positive and negative rates
        tp = prevalence * test["Sensitivity"]
        fn = prevalence * (1 - test["Sensitivity"])
        tn = (1 - prevalence) * test["Specificity"]
        fp = (1 - prevalence) * (1 - test["Specificity"])

        # Remove inconclusive portion and allocate later
        inconc = test["Inconclusive Rate"]
        tp *= (1 - inconc)
        fn *= (1 - inconc)
        tn *= (1 - inconc)
        fp *= (1 - inconc)

        # Base allocation from clear test results
        allocation = {
            "No CAD": tn,                  # correctly ruled out
            "CAD Undiagnosed": fn,         # missed disease
            # Among true positives, split into management strategies
            "CAD Untreated": 0.40 * tp,
            "PCI Treated": 0.45 * tp,
            "CABG Treated": 0.15 * tp,
            "Post-MI": 0.0,
            "Dead": 0.0
        }

        # Handle false positives (unnecessary follow-up)
        allocation["No CAD"] += 0.70 * fp
        allocation["CAD Untreated"] += 0.20 * fp
        allocation["PCI Treated"] += 0.10 * fp

        # Allocate inconclusive tests: assume some get diagnosed on retest
        allocation["CAD Untreated"] += 0.10 * inconc * prevalence
        allocation["CAD Undiagnosed"] += 0.90 * inconc * prevalence
        allocation["No CAD"] += inconc * (1 - prevalence)

        # Normalize to ensure proportions sum to 1
        total = sum(allocation.values())
        if total > 0:
            for state in allocation:
                allocation[state] /= total

        return allocation

    def calculate_initial_costs(self, strategy):
        """
        Compute expected initial costs for diagnostic test, ICA, interventions,
        and travel based on allocation proportions.
        """
        costs = self.params.costs
        test_cost = costs[strategy]
        travel_cost = costs["Travel Per Visit"][self.rural_distance]

        # Get allocation proportions from the decision tree
        alloc = self.classify_patients(strategy)

        # Calculate expected costs per patient
        init_costs = {}
        init_costs["Diagnostic Test"] = test_cost + travel_cost
        # All positive allocations go to ICA as part of workup
        positive_states = alloc["CAD Untreated"] + alloc["PCI Treated"] + alloc["CABG Treated"]
        init_costs["ICA"] = positive_states * costs["ICA"]
        init_costs["PCI"] = alloc["PCI Treated"] * costs["PCI"]
        init_costs["CABG"] = alloc["CABG Treated"] * costs["CABG"]
        # Some false positives still get unnecessary ICA
        init_costs["Unnecessary ICA"] = alloc["No CAD"] * 0.30 * costs["ICA"]
        # Two visits: initial and follow-up
        init_costs["Travel"] = travel_cost * 2

        return init_costs, alloc

# --------------------------------------------------------------------------------
# Markov model for long-term simulation
# --------------------------------------------------------------------------------
class MarkovModel:
    def __init__(self, params, initial_alloc, rural_distance="Distance 50-100 miles"):
        self.params = params
        self.cohort = initial_alloc.copy()  # start cohort proportions
        self.rural_distance = rural_distance

    def calculate_transition_matrix(self, age):
        """
        Build an N x N transition matrix for each health state,
        adjusting for age-specific mortality and treatment effects.
        """
        # Interpolate or lookup mortality for this age
        if age in self.params.age_mortality:
            mort = self.params.age_mortality[age]
        else:
            # linear interpolation between known ages
            ages = sorted(self.params.age_mortality)
            lower = max([a for a in ages if a <= age])
            upper = min([a for a in ages if a >= age])
            if lower == upper:
                mort = self.params.age_mortality[lower]
            else:
                m_low = self.params.age_mortality[lower]
                m_up = self.params.age_mortality[upper]
                mort = m_low + (m_up - m_low) * (age - lower) / (upper - lower)

        n = len(self.params.states)
        tm = np.zeros((n, n))
        idx = {state: i for i, state in enumerate(self.params.states)}
        p = self.params.base_transitions

        # For each state, fill transitions
        for state in self.params.states:
            i = idx[state]
            if state == "Dead":
                tm[i, idx["Dead"]] = 1.0
                continue

            # Everyone faces age-based mortality
            tm[i, idx["Dead"]] = mort

            if state == "No CAD":
                tm[i, idx["No CAD"]] = 1 - mort

            elif state == "CAD Undiagnosed":
                mi = p["MI Risk Untreated"]
                diag = 0.15
                adj_mort = mort * 1.2
                tm[i, idx["Dead"]] = adj_mort
                tm[i, idx["Post-MI"]] = (1 - adj_mort) * mi
                tm[i, idx["CAD Untreated"]] = (1 - adj_mort) * (1 - mi) * diag
                tm[i, idx["CAD Undiagnosed"]] = (1 - adj_mort) * (1 - mi) * (1 - diag)

            elif state == "CAD Untreated":
                med = self.params.treatment_effects["Medical Therapy"]
                mi = p["MI Risk Untreated"] * (1 - med["Reduction in MI Risk"])
                adj_mort = mort * (1 + 0.3 * (1 - med["Reduction in Mortality"]))
                pci_rate = 0.05
                tm[i, idx["Dead"]] = adj_mort
                tm[i, idx["Post-MI"]] = (1 - adj_mort) * mi
                tm[i, idx["PCI Treated"]] = (1 - adj_mort) * (1 - mi) * pci_rate
                tm[i, idx["CAD Untreated"]] = (1 - adj_mort) * (1 - mi) * (1 - pci_rate)

            elif state == "PCI Treated":
                eff = self.params.treatment_effects["PCI"]
                mi = p["MI Risk Treated"] * (1 - eff["Reduction in MI Risk"])
                adj_mort = mort * (1 + 0.2 * (1 - eff["Reduction in Mortality"]))
                tm[i, idx["Dead"]] = adj_mort
                tm[i, idx["Post-MI"]] = (1 - adj_mort) * mi
                tm[i, idx["PCI Treated"]] = (1 - adj_mort) * (1 - mi)

            elif state == "CABG Treated":
                eff = self.params.treatment_effects["CABG"]
                mi = p["MI Risk Treated"] * (1 - eff["Reduction in MI Risk"])
                adj_mort = mort * (1 + 0.15 * (1 - eff["Reduction in Mortality"]))
                cabg_repeat = self.params.repeat_intervention["CABG"]
                tm[i, idx["Dead"]] = adj_mort
                tm[i, idx["Post-MI"]] = (1 - adj_mort) * mi
                tm[i, idx["PCI Treated"]] = (1 - adj_mort) * (1 - mi) * cabg_repeat
                tm[i, idx["CABG Treated"]] = (1 - adj_mort) * (1 - mi) * (1 - cabg_repeat)

            elif state == "Post-MI":
                pm = p["Post-MI Mortality"]
                total_mort = mort + pm - mort * pm
                tm[i, idx["Dead"]] = total_mort
                tm[i, idx["Post-MI"]] = 1 - total_mort

        return tm

    def calculate_state_costs(self, cohort, age):
        """
        Compute annual costs for each health state including maintenance,
        repeat procedures, MI hospitalizations, and transport.
        """
        costs = self.params.costs
        visit_cost = costs["Travel Per Visit"][self.rural_distance]

        # State-specific yearly costs (medications, monitoring, follow-up)
        state_costs = {
            "No CAD": costs["CAD Monitoring"] * 0.2 + visit_cost * self.params.visits_per_year["No CAD"],
            "CAD Undiagnosed": 0,
            "CAD Untreated": costs["CAD Medication"] + costs["CAD Monitoring"] + visit_cost * self.params.visits_per_year["CAD Untreated"],
            "PCI Treated": costs["CAD Medication"] + costs["PCI Follow-up"] + visit_cost * self.params.visits_per_year["PCI Treated"],
            "CABG Treated": costs["CAD Medication"] + costs["CABG Follow-up"] + visit_cost * self.params.visits_per_year["CABG Treated"],
            "Post-MI": costs["Post-MI Care"] + visit_cost * self.params.visits_per_year["Post-MI"],
            "Dead": 0
        }

        # Calculate repeat intervention costs
        repeat_pci = self.params.repeat_intervention["PCI"] * cohort["PCI Treated"] * costs["PCI"]
        repeat_cabg = self.params.repeat_intervention["CABG"] * cohort["CABG Treated"] * costs["CABG"]

        # MI event costs (hospital and transport)
        mi_events = sum(cohort[state] * self.params.base_transitions["MI Risk Untreated"]
                        * (1 - self.params.treatment_effects["Medical Therapy"]["Reduction in MI Risk"])
                        for state in ["CAD Undiagnosed", "CAD Untreated"])
        mi_events += sum(cohort[state] * self.params.base_transitions["MI Risk Treated"]
                         * (1 - self.params.treatment_effects[state.split()[0]]["Reduction in MI Risk"])
                         for state in ["PCI Treated", "CABG Treated"])
        mi_hosp = mi_events * costs["MI Hospitalization"]
        mi_trans = mi_events * costs["Emergency Transport"][self.rural_distance]

        # Total annual costs for cycle
        annual_state_costs = sum(cohort[s] * state_costs[s] for s in self.params.states)
        total = annual_state_costs + repeat_pci + repeat_cabg + mi_hosp + mi_trans
        return total

    def calculate_state_qalys(self, cohort):
        """
        Compute annual QALYs for the cohort, adjusting for rural disutilities.
        """
        util = self.params.utilities
        disutil = self.params.rural_disutilities[self.rural_distance]
        qalys = 0
        for s in self.params.states:
            q = util[s] - (disutil if s != "Dead" else 0)
            qalys += cohort[s] * q
        return qalys

    def run_model(self):
        """
        Execute the Markov simulation over the time horizon.
        Tracks discounted costs and QALYs and returns totals and annual outputs.
        """
        total_costs = 0
        total_qalys = 0
        cohort = self.cohort.copy()
        results = []

        for t in range(self.params.time_horizon):
            age = self.params.starting_age + t
            tm = self.calculate_transition_matrix(age)

            # Calculate cost and QALYs this cycle
            cost = self.calculate_state_costs(cohort, age)
            qaly = self.calculate_state_qalys(cohort)

            # Discount
            df = 1 / (1 + self.params.discount_rate) ** t
            total_costs += cost * df
            total_qalys += qaly * df

            # Record cycle results
            results.append({"Year": t, "Age": age, "Cost": cost * df, "QALY": qaly * df})

            # Transition cohort to next cycle
            next_cohort = {s: 0 for s in self.params.states}
            for i, from_s in enumerate(self.params.states):
                for j, to_s in enumerate(self.params.states):
                    next_cohort[to_s] += cohort[from_s] * tm[i, j]
            cohort = next_cohort

        return total_qalys, total_costs, results

# --------------------------------------------------------------------------------
# Cost-effectiveness analysis wrapper
# --------------------------------------------------------------------------------
class CostEffectivenessAnalysis:
    """
    Runs and compares multiple strategies: initial decision tree, Markov model,
    computes ICERs, and supports sensitivity analyses.
    """
    def __init__(self, params):
        self.params = params

    def run_analysis(self, strategies, rural_distance="Distance 50-100 miles"):
        results = {}
        for strat in strategies:
            dp = DiagnosticPathway(self.params, rural_distance)
            init_costs, alloc = dp.calculate_initial_costs(strat)
            mm = MarkovModel(self.params, alloc, rural_distance)
            qalys, long_costs, annual = mm.run_model()
            total_init = sum(init_costs.values())
            results[strat] = {
                "QALYs": qalys,
                "Costs": total_init + long_costs,
                "Initial Costs": init_costs,
                "Annual" : annual
            }
        return results

    def calculate_icer(self, results, ref):
        ref_res = results[ref]
        icers = {}
        for strat, res in results.items():
            if strat == ref: continue
            dc = res["Costs"] - ref_res["Costs"]
            dq = res["QALYs"] - ref_res["QALYs"]
            icers[strat] = dc / dq if dq > 0 else ("Dominated" if dc > 0 else "Dominates")
        return icers

    # Sensitivity methods omitted for brevity; add detailed comments similarly

# --------------------------------------------------------------------------------
# Entry point to run all scenarios
# --------------------------------------------------------------------------------
def run_cost_effectiveness_model():
    params = ModelParams()
    cea = CostEffectivenessAnalysis(params)
    strategies = ["FFRCT", "Stress Test"]
    scenarios = ["Distance < 50 miles", "Distance 50-100 miles", "Distance > 100 miles"]
    all_results = {}
    for sc in scenarios:
        res = cea.run_analysis(strategies, sc)
        ic = cea.calculate_icer(res, "Stress Test")
        all_results[sc] = {"Results": res, "ICERs": ic}
    return all_results
