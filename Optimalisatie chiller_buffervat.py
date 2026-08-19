import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. DATA INLEZEN EN VOORBEREIDEN
# -----------------------------------------------------------------------------
CSV_PATH = "plant1.csv"

try:
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
except FileNotFoundError:
    print(
        f"Waarschuwing: Bestand '{CSV_PATH}' niet gevonden. Zorg dat het"
        " aanwezig is."
    )

N_STEPS = 96  # eerste 24 uur = 96 kwartieren
df = df.iloc[:N_STEPS].reset_index(drop=True)

dt = 0.25  # uur

price = df["price"].values / 1000.0  # EUR/kWh (spotprijs)
P_pv = df["pv_production"].values  # kW
P_inflex = df["inflex_load"].values  # kW
T_outdoor = df["outdoor_temperature"].values  # degC

time_h = np.arange(N_STEPS) * dt

# -----------------------------------------------------------------------------
# 2. PARAMETERS & KOSTENCONFIGURATIE
# -----------------------------------------------------------------------------
Q_demand = np.full(N_STEPS, 45.0)  # kW thermisch, constante proceskoeling
P_max_chiller = 250.0  # kW elektrisch

temp_points = [0, 5, 10, 15, 20, 25, 30]
cop_points = [4.5, 4.3, 4.0, 3.7, 3.4, 3.1, 2.8]
COP_t = np.interp(T_outdoor, temp_points, cop_points)

V_buffer = 10.0  # m3
KWH_PER_M3 = 8.0
E_therm_max = V_buffer * KWH_PER_M3
E_therm_min = 0.0
eta_loss = 0.998
E_therm_0 = 0.5 * E_therm_max

# Uitgebreide kostenparameters
cost_params = {
    "fixed_cost_per_kwh": 0.05,  # Vaste distributiekosten per kWh
    "net_loss_factor": 1.02,  # Netverliesfactor (bijv. 2% extra)
    "access_power": 50.0,  # Gecontracteerd vermogen (kW) - Access Power
    "excess_price": 2.0,  # Vaste prijs per kW Access Power
    "peak_price": 1.5,  # Prijs per kW gemeten piekbelasting
    "penalty_price": 5.0,  # Boete per kW boven access power
}

# -----------------------------------------------------------------------------
# 3. OPTIMALISATIE & BASELINE
# -----------------------------------------------------------------------------


def solve_optimized():
    P_chiller = cp.Variable(N_STEPS, nonneg=True)
    P_grid_import = cp.Variable(N_STEPS, nonneg=True)
    E_therm = cp.Variable(N_STEPS + 1)

    # Variabelen voor piekbelasting
    P_peak = cp.Variable(nonneg=True)
    P_peak_above = cp.Variable(nonneg=True)

    Q_chiller = cp.multiply(COP_t, P_chiller)

    constraints = [
        P_chiller <= P_max_chiller,
        E_therm[0] == E_therm_0,
        E_therm >= E_therm_min,
        E_therm <= E_therm_max,
        E_therm[N_STEPS] == E_therm_0,
        # Vermogensbalans (zonder export)
        P_pv + P_grid_import == P_chiller + P_inflex,
        # Pieklogica
        P_peak >= cp.max(P_grid_import),
        P_peak_above >= P_peak - cost_params["access_power"],
        P_peak_above >= 0,
    ]

    # Vectoriële thermische balans
    constraints += [
        E_therm[1:] == eta_loss * E_therm[:-1] + (Q_chiller - Q_demand) * dt
    ]

    # 1. Spot- en energiekosten: (spot * netverlies + vaste kosten) * verbruik * dt
    energy_cost_per_kwh = (
        price * cost_params["net_loss_factor"]
    ) + cost_params["fixed_cost_per_kwh"]
    energy_cost = cp.sum(cp.multiply(P_grid_import, energy_cost_per_kwh) * dt)

    # 2. Capaciteits- en piekkosten
    capacity_cost = (
        (cost_params["access_power"] * cost_params["excess_price"])
        + (P_peak * cost_params["peak_price"])
        + (P_peak_above * cost_params["penalty_price"])
    )

    problem = cp.Problem(cp.Minimize(energy_cost + capacity_cost), constraints)
    problem.solve(solver=cp.HIGHS)

    return {
        "P_chiller": P_chiller.value,
        "P_grid_import": P_grid_import.value,
        "E_therm": E_therm.value,
        "energy_cost": energy_cost.value,
        "capacity_cost": capacity_cost.value,
        "cost": problem.value,
    }


def solve_baseline():
    P_chiller_bl = Q_demand / COP_t  # Chiller volgt vraag direct
    net_bl = P_chiller_bl + P_inflex - P_pv
    P_grid_import_bl = np.maximum(net_bl, 0.0)

    E_therm_bl = np.full(
        N_STEPS + 1, E_therm_0
    )  # Buffervat blijft op startniveau

    P_peak_bl = P_grid_import_bl.max()
    P_peak_above_bl = max(0.0, P_peak_bl - cost_params["access_power"])

    # Kostenberekening baseline
    energy_cost_per_kwh_bl = (
        price * cost_params["net_loss_factor"]
    ) + cost_params["fixed_cost_per_kwh"]
    energy_cost_bl = np.sum(P_grid_import_bl * energy_cost_per_kwh_bl * dt)

    capacity_cost_bl = (
        (cost_params["access_power"] * cost_params["excess_price"])
        + (P_peak_bl * cost_params["peak_price"])
        + (P_peak_above_bl * cost_params["penalty_price"])
    )

    return {
        "P_chiller": P_chiller_bl,
        "P_grid_import": P_grid_import_bl,
        "E_therm": E_therm_bl,
        "energy_cost": energy_cost_bl,
        "capacity_cost": capacity_cost_bl,
        "cost": energy_cost_bl + capacity_cost_bl,
    }


opt = solve_optimized()
base = solve_baseline()

# -----------------------------------------------------------------------------
# 4. KOSTENVERGELIJKING
# -----------------------------------------------------------------------------
savings_pct = 100.0 * (base["cost"] - opt["cost"]) / base["cost"]

print(f"{'':25}{'Baseline':>12}{'Geoptimaliseerd':>18}")
print("-" * 55)
print(
    f"{'Energiekosten [EUR]':25}{base['energy_cost']:>12.2f}{opt['energy_cost']:>18.2f}"
)
print(
    f"{'Capaciteitskosten [EUR]':25}{base['capacity_cost']:>12.2f}{opt['capacity_cost']:>18.2f}"
)
print("-" * 55)
print(f"{'Totale kosten [EUR]':25}{base['cost']:>12.2f}{opt['cost']:>18.2f}")
print("-" * 55)
print(f"{'Totale kostenbesparing':25}{'':>12}{savings_pct:>17.1f}%")

# -----------------------------------------------------------------------------
# 5. VISUALISATIES EN GRAFIEKEN
# -----------------------------------------------------------------------------

# --- Grafiek 1: Elektrisch vermogen + prijs ---
fig1, ax1 = plt.subplots(figsize=(13, 5))
ax1.step(
    time_h,
    opt["P_chiller"],
    where="post",
    label="P_chiller (geoptimaliseerd)",
    color="tab:blue",
    linewidth=2,
)
ax1.step(
    time_h,
    base["P_chiller"],
    where="post",
    label="P_chiller (baseline)",
    color="tab:blue",
    linestyle=":",
    alpha=0.6,
    linewidth=1.5,
)
ax1.step(
    time_h, P_pv, where="post", label="P_PV", color="tab:orange", linewidth=1.5
)
ax1.step(
    time_h,
    opt["P_grid_import"],
    where="post",
    label="P_grid_import (geopt)",
    color="tab:green",
    linewidth=1.5,
)
ax1.set_xlabel("Tijd [uur]")
ax1.set_ylabel("Elektrisch vermogen [kW]")
ax1.set_xlim(0, 24)
ax1.grid(alpha=0.3)
ax1.legend(loc="upper left")

ax1b = ax1.twinx()
ax1b.step(
    time_h,
    price,
    where="post",
    color="black",
    linewidth=1.2,
    alpha=0.6,
    label="Elektriciteitsprijs",
)
ax1b.set_ylabel("Elektriciteitsprijs [EUR/kWh]")
ax1b.legend(loc="upper right")
ax1.set_title(
    f"Grafiek 1 - Elektrisch vermogen en prijs (buffervat {V_buffer} m3 /"
    f" {E_therm_max:.0f} kWh)"
)
fig1.tight_layout()
plt.show()

# --- Grafiek 2: Laadtoestand buffervat ---
fig2, ax2 = plt.subplots(figsize=(13, 4.5))
time_h_full = np.arange(N_STEPS + 1) * dt
ax2.plot(
    time_h_full,
    opt["E_therm"],
    label="E_therm (geoptimaliseerd)",
    color="tab:blue",
    linewidth=2,
)
ax2.plot(
    time_h_full,
    base["E_therm"],
    label="E_therm (baseline)",
    color="tab:gray",
    linestyle=":",
    linewidth=1.5,
)
ax2.axhline(
    E_therm_max, color="red", linestyle=":", linewidth=1, label="E_therm_max"
)
ax2.axhline(
    E_therm_min, color="red", linestyle=":", linewidth=1, label="E_therm_min"
)
ax2.set_xlabel("Tijd [uur]")
ax2.set_ylabel("Thermische inhoud buffervat [kWh]")
ax2.set_xlim(0, 24)
ax2.set_ylim(-1, E_therm_max + 5)
ax2.grid(alpha=0.3)
ax2.legend(loc="best")
ax2.set_title(f"Grafiek 2 - Laadtoestand koudwaterbuffervat ({V_buffer} m3)")
fig2.tight_layout()
plt.show()