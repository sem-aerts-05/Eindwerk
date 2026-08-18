import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. DATA INLEZEN
# ==========================================
df = pd.read_csv("plant1.csv", parse_dates=["timestamp"])

# ==========================================
# 2. PRIJS EN KOELVRAAG ANALYSEREN
# ==========================================
print("=== STATISTIEKEN ELEKTRICITEITSPRIJS ===")
print(df["price"].describe())

print("\n=== STATISTIEKEN THERMISCHE KOELVRAAG (thermal_load) ===")
print(df["thermal_load"].describe())

# ==========================================
# 3. VISUALISATIE (Plotten van de koelvraag)
# ==========================================
# We nemen hier de eerste 7 dagen (7 dagen * 96 kwartieren = 672 stappen) 
# zodat de grafiek overzichtelijk en leesbaar blijft.
df_subset = df.iloc[: 96 * 7]

plt.figure(figsize=(12, 5))
plt.plot(
    df_subset["timestamp"],
    df_subset["thermal_load"],
    label="Totale Koelvraag (thermal_load)",
    color="tab:cyan",
    linewidth=1.5
)

plt.title("Totale Koelvraag van de Fabriek (Eerste 7 dagen)", fontsize=14, fontweight="bold")
plt.xlabel("Tijdstip", fontsize=11)
plt.ylabel("Thermisch vermogen [kW]", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()

# Sla de grafiek direct op als afbeelding
plt.savefig("koelvraag_overzicht.png", dpi=150)
plt.show()
print("\nGrafiek succesvol opgeslagen als 'koelvraag_overzicht.png'.")