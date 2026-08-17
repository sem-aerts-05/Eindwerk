"""
vlaamse_energiefactuur.py

Objectgeoriënteerde berekening van de Vlaamse energiefactuur.

Deze module volgt de rekenlogica uit het aangeleverde notebook:
- P1, P2, HODB, Aerotrim, Kallo, SAPAC en STC vormen samen het totale
  energievolume voor de bepaling van de gewogen energieprijs.
- P1 wordt daarna gebruikt voor het gefactureerde volume, maandpiek en
  reactieve energie.
- Spotprijzen worden per uur gekoppeld aan de kwartiermetingen.
- Forwardpositie = 1,3 MW, dus 325 kWh per kwartier.
- De tarieven hieronder zijn de waarden uit het notebook en kunnen via
  InvoiceConfig worden aangepast.

Vereisten:
    pip install pandas numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_dutch_number(value: float, decimals: int = 2) -> str:
    """Belgische/Nederlandse getalnotatie: 1234.56 -> 1.234,56."""
    if value is None or pd.isna(value):
        return ""

    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def format_euro(value: float, decimals: int = 2) -> str:
    """Formatteert een bedrag als bijvoorbeeld € 110.525,20."""
    sign = "-" if value < 0 else ""
    return f"{sign}€ {format_dutch_number(abs(value), decimals)}"


def format_unit_price(value: float, decimals: int = 3,
                      unit: str = "€/MWh") -> str:
    """Formatteert een eenheidsprijs."""
    return f"{format_dutch_number(value, decimals)} {unit}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InvoiceConfig:
    """
    Alle factuurparameters.

    De standaardwaarden zijn de waarden uit het aangeleverde notebook.
    """

    # Energie
    forward_volume_mw: float = 1.3
    forward_price_eur_mwh: float = 85.58
    onbalans_eur_mwh: float = 4.612
    go_eur_mwh: float = 3.04

    # WKC / GSC
    wkc_eur_mwh: float = 3.556
    gsc_eur_mwh: float = 10.835
    wkc_discount_pct: float = 0.47
    gsc_discount_pct: float = 0.47

    # Netverlies
    net_loss_pct: float = 1.85

    # Nettarieven
    capacity_annual_eur_kw: float = 46.8630636
    data_management_annual_eur: float = 57.65
    reactive_eur_mvarh: float = 13.4149

    # Distributie variabel
    access_power_annual_eur_kw: float = 31.5677340
    access_power_p1_kw: float = 2850.0
    historical_penalty_eur_kw: float = 71.0
    overshoot_multiplier: float = 1.5

    # Afnametarief
    odv_eur_mwh: float = 4.0283
    surcharges_eur_mwh: float = 0.1922

    # Belastingen
    special_excise_eur_mwh: float = 10.69
    energy_fund_eur_month: float = 189.48

    # BTW: het notebook vermeldt 21% op de energieregels.
    vat_percent: float = 21.0


# ---------------------------------------------------------------------------
# Meter reader
# ---------------------------------------------------------------------------

class MeterDataReader:
    """
    Robuuste CSV-reader voor Belgische energiemeterbestanden.

    Ondersteunt onder andere:
      - puntkomma + decimale komma;
      - komma + decimale punt;
      - UTF-8 BOM;
      - registers 'Afname Actief' en 'Afname Reactief';
      - de kolommen 'Van (datum)' en 'Van (tijdstip)';
      - reeds samengestelde datasets met 'timestamp'/'Price'.
    """

    ACTIVE_REGISTER = "Afname Actief"
    REACTIVE_REGISTER = "Afname Reactief"

    @staticmethod
    def _normalise_column_name(name: object) -> str:
        return str(name).strip().replace("\ufeff", "")

    @classmethod
    def _read_raw(cls, path: PathLike) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"CSV-bestand niet gevonden: {path.resolve()}"
            )

        # Eerst autodetectie via Python's csv-engine.
        attempts = [
            {"sep": ";", "decimal": ",", "encoding": "utf-8-sig"},
            {"sep": ",", "decimal": ".", "encoding": "utf-8-sig"},
            {"sep": ";", "decimal": ".", "encoding": "utf-8-sig"},
            {"sep": ",", "decimal": ",", "encoding": "utf-8-sig"},
        ]

        best = None
        best_score = -1

        for kwargs in attempts:
            try:
                df = pd.read_csv(path, **kwargs)
                score = len(df.columns)
                if score > best_score:
                    best = df
                    best_score = score
            except Exception:
                continue

        if best is None:
            raise ValueError(f"Kan CSV niet lezen: {path}")

        best.columns = [
            cls._normalise_column_name(c) for c in best.columns
        ]
        return best

    @staticmethod
    def _numeric(series: pd.Series) -> pd.Series:
        """
        Zet zowel 1.234,56 als 1234.56 om naar float.

        Voor waarden met zowel punt als komma wordt de laatste separator
        beschouwd als decimaalteken.
        """
        def convert(value):
            if pd.isna(value):
                return np.nan

            s = str(value).strip().replace("\u00a0", "").replace(" ", "")
            if not s:
                return np.nan

            if "," in s and "." in s:
                if s.rfind(",") > s.rfind("."):
                    s = s.replace(".", "").replace(",", ".")
                else:
                    s = s.replace(",", "")
            elif "," in s:
                s = s.replace(",", ".")

            try:
                return float(s)
            except ValueError:
                return np.nan

        return series.map(convert)

    @staticmethod
    def _datetime_from_meter(df: pd.DataFrame) -> pd.Series:
        if "Datetime" in df.columns:
            return pd.to_datetime(df["Datetime"], errors="coerce")

        if "datetime" in df.columns:
            return pd.to_datetime(df["datetime"], errors="coerce")

        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"], errors="coerce")

        required = {"Van (datum)", "Van (tijdstip)"}
        if required.issubset(df.columns):
            text = (
                df["Van (datum)"].astype(str).str.strip()
                + " "
                + df["Van (tijdstip)"].astype(str).str.strip()
            )
            return pd.to_datetime(
                text,
                dayfirst=True,
                errors="coerce",
            )

        raise ValueError(
            "Geen tijdstempel gevonden. Verwacht 'timestamp'/'Datetime' "
            "of 'Van (datum)' + 'Van (tijdstip)'."
        )

    @classmethod
    def read_meter_csv(
        cls,
        path: PathLike,
        meter_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Leest een meterbestand.

        Resultaat bevat minimaal:
            Datetime
            Volume [kWh]
            Register

        Als Register aanwezig is, blijven alle registers behouden.
        Gebruik filter_register() om specifiek actief/reactief te kiezen.
        """
        df = cls._read_raw(path).copy()

        # Volume
        volume_column = next(
            (
                c for c in df.columns
                if c.strip().lower() in {
                    "volume",
                    "volume [kwh]",
                    "volume [kwh] ",
                }
            ),
            None,
        )

        # Sommige exports gebruiken een andere hoofdlettercombinatie.
        if volume_column is None:
            for c in df.columns:
                if "volume" in c.lower():
                    volume_column = c
                    break

        if volume_column is None:
            raise ValueError(
                f"Geen volumekolom gevonden in {path}. "
                f"Kolommen: {list(df.columns)}"
            )

        df["Volume [kWh]"] = cls._numeric(df[volume_column])

        if "Register" not in df.columns:
            df["Register"] = cls.ACTIVE_REGISTER

        df["Register"] = df["Register"].astype(str).str.strip()
        df["Datetime"] = cls._datetime_from_meter(df)

        if meter_name is None:
            meter_name = Path(path).stem

        df["Meter"] = meter_name

        result = df[
            ["Datetime", "Meter", "Register", "Volume [kWh]"]
        ].copy()

        result = result.dropna(subset=["Datetime"])
        result = result.sort_values("Datetime").reset_index(drop=True)

        return result

    @classmethod
    def read_active_csv(
        cls,
        path: PathLike,
        meter_name: Optional[str] = None,
    ) -> pd.DataFrame:
        df = cls.read_meter_csv(path, meter_name)
        return df[
            df["Register"].str.casefold()
            == cls.ACTIVE_REGISTER.casefold()
        ].copy()

    @classmethod
    def read_reactive_csv(
        cls,
        path: PathLike,
        meter_name: Optional[str] = None,
    ) -> pd.DataFrame:
        df = cls.read_meter_csv(path, meter_name)
        return df[
            df["Register"].str.casefold()
            == cls.REACTIVE_REGISTER.casefold()
        ].copy()

    @classmethod
    def load_spot_prices(cls, path: PathLike) -> pd.DataFrame:
        """
        Leest een spotprijsbestand.

        Ondersteunt bijvoorbeeld:
            timestamp,Price
            2026-06-01 00:00:00,80.12

        maar ook:
            Datetime,Price

        Als de spotdata geen timestamp bevat maar alleen Price heeft,
        worden de prijzen later via kwartier-volgorde herhaald indien
        exact één maand/jaar op de P1-data wordt gebruikt.
        """
        df = cls._read_raw(path).copy()

        time_col = next(
            (
                c for c in df.columns
                if c.lower() in {"timestamp", "datetime", "date", "tijd"}
            ),
            None,
        )

        price_col = next(
            (
                c for c in df.columns
                if c.lower() in {
                    "price",
                    "spotprijs",
                    "spotprijs [€/mwh]",
                    "price [€/mwh]",
                }
            ),
            None,
        )

        if price_col is None:
            for c in df.columns:
                if "price" in c.lower() or "spot" in c.lower():
                    price_col = c
                    break

        if price_col is None:
            raise ValueError(
                f"Geen spotprijskolom gevonden in {path}. "
                f"Kolommen: {list(df.columns)}"
            )

        result = pd.DataFrame()
        result["Spotprijs [€/MWh]"] = cls._numeric(df[price_col])

        if time_col is not None:
            result["Datetime"] = pd.to_datetime(
                df[time_col], dayfirst=True, errors="coerce"
            )
            result = result.dropna(subset=["Datetime"])
            result = (
                result.sort_values("Datetime")
                .drop_duplicates("Datetime", keep="last")
                .reset_index(drop=True)
            )

        return result

    @classmethod
    def combine_meters(
        cls,
        meter_paths: Mapping[str, PathLike],
    ) -> pd.DataFrame:
        """
        Combineert P1, P2, HODB, Aerotrim, Kallo, SAPAC en STC
        op timestamp.

        De namen in meter_paths bepalen de kolomnamen.
        """
        frames = {}

        for meter_name, path in meter_paths.items():
            active = cls.read_active_csv(path, meter_name)
            if active.empty:
                raise ValueError(
                    f"Geen 'Afname Actief' records gevonden voor {meter_name}."
                )

            active = active[["Datetime", "Volume [kWh]"]].rename(
                columns={"Volume [kWh]": f"Volume_{meter_name} [kWh]"}
            )
            frames[meter_name] = active

        # P1 is de referentietijdas, net als in het notebook.
        if "P1" not in frames:
            raise ValueError("P1 is verplicht als referentiemeter.")

        combined = frames["P1"].copy()

        for name, frame in frames.items():
            if name == "P1":
                continue

            combined = combined.merge(
                frame,
                on="Datetime",
                how="inner",
                validate="one_to_one",
            )

        volume_cols = [
            c for c in combined.columns
            if c.startswith("Volume_") and c.endswith("[kWh]")
        ]

        combined["Totaalvolume [kWh]"] = combined[volume_cols].sum(axis=1)

        return combined.sort_values("Datetime").reset_index(drop=True)

    @classmethod
    def attach_spot_prices(
        cls,
        combined: pd.DataFrame,
        spot: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Koppelt uurlijkse spotprijzen aan kwartiermetingen.

        Exacte timestamp -> forward-fill via merge_asof.
        Bij een spotbestand zonder timestamp wordt de Price-kolom viermaal
        per uur herhaald, op basis van de volgorde van de kwartierdata.
        """
        result = combined.copy().sort_values("Datetime").reset_index(drop=True)
        spot = spot.copy()

        if "Datetime" in spot.columns:
            spot = spot.sort_values("Datetime").reset_index(drop=True)

            result = pd.merge_asof(
                result,
                spot[["Datetime", "Spotprijs [€/MWh]"]],
                on="Datetime",
                direction="backward",
                tolerance=pd.Timedelta(hours=1),
            )
        else:
            prices = spot["Spotprijs [€/MWh]"].dropna().to_numpy()

            if len(prices) == 0:
                raise ValueError("Spotprijsbestand bevat geen prijzen.")

            expected_hours = int(np.ceil(len(result) / 4))
            if len(prices) < expected_hours:
                raise ValueError(
                    f"Te weinig spotprijzen: {len(prices)} prijzen voor "
                    f"{expected_hours} benodigde uren."
                )

            repeated = np.repeat(prices[:expected_hours], 4)
            result["Spotprijs [€/MWh]"] = repeated[:len(result)]

        if result["Spotprijs [€/MWh]"].isna().any():
            missing = int(result["Spotprijs [€/MWh]"].isna().sum())
            raise ValueError(
                f"{missing} kwartierpunten hebben geen spotprijs."
            )

        return result


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class InvoiceResult:
    invoice: pd.DataFrame
    detailed_data: pd.DataFrame
    metrics: Dict[str, float]

    def __str__(self) -> str:
        return self.to_text()

    def to_text(self) -> str:
        lines = []
        lines.append("=" * 92)
        lines.append("VLAAMSE ENERGIEFACTUUR")
        lines.append("=" * 92)

        for _, row in self.invoice.iterrows():
            description = str(row["Omschrijving"])
            total = row["Totaal excl. BTW"]

            if description.startswith("Totaal") or description.startswith(
                "Subtotaal"
            ):
                lines.append("-" * 92)

            lines.append(
                f"{description:<42} "
                f"{str(total):>24}"
            )

        lines.append("=" * 92)

        lines.append(
            f"Totaal volume portefeuille : "
            f"{format_dutch_number(self.metrics['total_volume_mwh'], 3)} MWh"
        )
        lines.append(
            f"Spotvolume                 : "
            f"{format_dutch_number(self.metrics['spot_volume_mwh'], 3)} MWh"
        )
        lines.append(
            f"Forwardvolume              : "
            f"{format_dutch_number(self.metrics['forward_volume_mwh'], 3)} MWh"
        )
        lines.append(
            f"Gewogen energieprijs       : "
            f"{format_unit_price(self.metrics['average_energy_price'], 3)}"
        )
        lines.append(
            f"P1 factuurvolume           : "
            f"{format_dutch_number(self.metrics['p1_volume_mwh'], 3)} MWh"
        )
        lines.append(
            f"P1 maandpiek               : "
            f"{format_dutch_number(self.metrics['month_peak_kw'], 0)} kW"
        )
        lines.append("=" * 92)

        return "\n".join(lines)

    def to_excel(self, path: PathLike) -> None:
        self.invoice.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class VlaamseEnergiefactuurCalculator:
    """Hoofdcalculator voor de Vlaamse energiefactuur."""

    def __init__(
        self,
        config: InvoiceConfig,
        data: pd.DataFrame,
        p1_full_data: pd.DataFrame,
    ):
        self.config = config
        self.data = data.copy()
        self.p1_full_data = p1_full_data.copy()

        if self.data.empty:
            raise ValueError("Geen meetdata beschikbaar.")

        self.data = self.data.sort_values("Datetime").reset_index(drop=True)

        self.data["Totaalvolume [kWh]"] = pd.to_numeric(
            self.data["Totaalvolume [kWh]"], errors="coerce"
        )

        if "Spotprijs [€/MWh]" not in self.data.columns:
            raise ValueError("Spotprijzen ontbreken in de meetdata.")

        self.data["Spotprijs [€/MWh]"] = pd.to_numeric(
            self.data["Spotprijs [€/MWh]"], errors="coerce"
        )

    # ------------------------------------------------------------------
    # Basisstatistieken
    # ------------------------------------------------------------------

    @property
    def period_start(self) -> pd.Timestamp:
        return self.data["Datetime"].min()

    @property
    def period_end(self) -> pd.Timestamp:
        return self.data["Datetime"].max()

    @property
    def year(self) -> int:
        return int(self.period_start.year)

    @property
    def month(self) -> int:
        return int(self.period_start.month)

    @property
    def days_in_month(self) -> int:
        return int(self.period_start.days_in_month)

    @property
    def days_in_year(self) -> int:
        return int(
            pd.Timestamp(self.year, 12, 31).dayofyear
        )

    @property
    def period_label(self) -> str:
        return (
            f"{self.period_start:%d-%m-%Y} - "
            f"{self.period_end:%d-%m-%Y}"
        )

    def _p1_active(self) -> pd.DataFrame:
        return self.p1_full_data[
            self.p1_full_data["Register"].str.casefold()
            == MeterDataReader.ACTIVE_REGISTER.casefold()
        ].copy()

    def _p1_reactive(self) -> pd.DataFrame:
        return self.p1_full_data[
            self.p1_full_data["Register"].str.casefold()
            == MeterDataReader.REACTIVE_REGISTER.casefold()
        ].copy()

    # ------------------------------------------------------------------
    # Energie
    # ------------------------------------------------------------------

    def calculate_energy(self) -> Dict[str, float]:
        c = self.config
        df = self.data.copy()

        # Exact dezelfde kwartierformule als in het notebook:
        # 1,3 MW / 4 * 1000 = 325 kWh per kwartier.
        forward_kwh_per_quarter = (c.forward_volume_mw / 4.0) * 1000.0

        df["Forward_volume [kWh]"] = forward_kwh_per_quarter
        df["Volume_spot [kWh]"] = (
            df["Totaalvolume [kWh]"]
            - df["Forward_volume [kWh]"]
        )

        # Kosten per kwartier
        df["Kosten_spot [€]"] = (
            df["Spotprijs [€/MWh]"]
            * df["Volume_spot [kWh]"]
            / 1000.0
        )

        df["Kosten_forward [€]"] = (
            c.forward_price_eur_mwh
            * df["Forward_volume [kWh]"]
            / 1000.0
        )

        df["Kosten_onbalans [€]"] = (
            c.onbalans_eur_mwh
            * df["Totaalvolume [kWh]"]
            / 1000.0
        )

        df["Kosten_GO [€]"] = (
            c.go_eur_mwh
            * df["Totaalvolume [kWh]"]
            / 1000.0
        )

        df["Kosten_totaal_portefeuille [€]"] = (
            df["Kosten_spot [€]"]
            + df["Kosten_forward [€]"]
            + df["Kosten_onbalans [€]"]
            + df["Kosten_GO [€]"]
        )

        total_volume_kwh = float(df["Totaalvolume [kWh]"].sum())
        forward_kwh = float(df["Forward_volume [kWh]"].sum())
        spot_kwh = float(df["Volume_spot [kWh]"].sum())
        total_cost = float(
            df["Kosten_totaal_portefeuille [€]"].sum()
        )

        average_price = (
            total_cost / (total_volume_kwh / 1000.0)
            if total_volume_kwh
            else 0.0
        )

        # P1 factuurvolume
        p1 = self._p1_active()
        p1_volume_kwh = float(p1["Volume [kWh]"].sum())
        p1_volume_mwh = p1_volume_kwh / 1000.0

        # Exacte notebook-formule:
        # netverliesprijs = gemiddelde energieprijs * 1,85%
        net_loss_price = average_price * (
            c.net_loss_pct / 100.0
        )

        energy_cost = p1_volume_mwh * average_price
        wkc_cost = p1_volume_mwh * c.wkc_eur_mwh
        gsc_cost = p1_volume_mwh * c.gsc_eur_mwh

        wkc_discount = (
            p1_volume_mwh
            * c.wkc_eur_mwh
            * (c.wkc_discount_pct)
        )

        gsc_discount = (
            p1_volume_mwh
            * c.gsc_eur_mwh
            * (c.gsc_discount_pct)
        )

        net_loss_cost = p1_volume_mwh * net_loss_price

        total_energy_cost = (
            energy_cost
            + wkc_cost
            - wkc_discount
            + gsc_cost
            - gsc_discount
            + net_loss_cost
        )

        self.data = df

        return {
            "total_volume_kwh": total_volume_kwh,
            "total_volume_mwh": total_volume_kwh / 1000.0,
            "forward_volume_kwh": forward_kwh,
            "forward_volume_mwh": forward_kwh / 1000.0,
            "spot_volume_kwh": spot_kwh,
            "spot_volume_mwh": spot_kwh / 1000.0,
            "spot_cost": float(df["Kosten_spot [€]"].sum()),
            "forward_cost": float(df["Kosten_forward [€]"].sum()),
            "onbalans_cost": float(df["Kosten_onbalans [€]"].sum()),
            "go_cost": float(df["Kosten_GO [€]"].sum()),
            "portfolio_total_cost": total_cost,
            "average_energy_price": average_price,
            "p1_volume_kwh": p1_volume_kwh,
            "p1_volume_mwh": p1_volume_mwh,
            "energy_cost": energy_cost,
            "wkc_cost": wkc_cost,
            "wkc_discount": wkc_discount,
            "gsc_cost": gsc_cost,
            "gsc_discount": gsc_discount,
            "net_loss_price": net_loss_price,
            "net_loss_cost": net_loss_cost,
            "total_energy_cost": total_energy_cost,
        }

    # ------------------------------------------------------------------
    # Nettarieven
    # ------------------------------------------------------------------

    def calculate_grid(self, energy: Dict[str, float]) -> Dict[str, float]:
        c = self.config

        p1_active = self._p1_active()
        p1_reactive = self._p1_reactive()

        # Databeheer
        data_management_day = (
            c.data_management_annual_eur / self.days_in_year
        )
        data_management_month = (
            data_management_day * self.days_in_month
        )

        # Reactief
        reactive_kvarh = float(
            p1_reactive["Volume [kWh]"].sum()
        )
        reactive_mvarh = reactive_kvarh / 1000.0
        reactive_cost = reactive_mvarh * c.reactive_eur_mvarh

        # Maandpiek P1
        month_peak_kw = (
            float(p1_active["Volume [kWh]"].max()) * 4.0
            if not p1_active.empty
            else 0.0
        )

        # Capaciteitstarief
        capacity_monthly_eur_kw = (
            c.capacity_annual_eur_kw
            / self.days_in_year
            * self.days_in_month
        )
        capacity_cost = (
            capacity_monthly_eur_kw * month_peak_kw
        )

        # Distributie variabel
        access_power_monthly_eur_kw = (
            c.access_power_annual_eur_kw
            / self.days_in_year
            * self.days_in_month
        )

        access_cost = (
            access_power_monthly_eur_kw
            * c.access_power_p1_kw
        )

        # Exacte formule uit het notebook.
        historical_penalty = (
            c.historical_penalty_eur_kw
            * access_power_monthly_eur_kw
            * c.overshoot_multiplier
        )

        overshoot_kw = max(
            0.0,
            month_peak_kw - c.access_power_p1_kw,
        )

        distribution_variable = access_cost + historical_penalty

        # Afnametarief
        odv_cost = (
            energy["p1_volume_mwh"]
            * c.odv_eur_mwh
        )

        surcharges_cost = (
            energy["p1_volume_mwh"]
            * c.surcharges_eur_mwh
        )

        withdrawal_tariff = odv_cost + surcharges_cost

        total_grid = (
            data_management_month
            + reactive_cost
            + capacity_cost
            + distribution_variable
            + withdrawal_tariff
        )

        return {
            "data_management_day": data_management_day,
            "data_management_month": data_management_month,
            "reactive_mvarh": reactive_mvarh,
            "reactive_cost": reactive_cost,
            "month_peak_kw": month_peak_kw,
            "capacity_monthly_eur_kw": capacity_monthly_eur_kw,
            "capacity_cost": capacity_cost,
            "access_power_monthly_eur_kw": access_power_monthly_eur_kw,
            "access_cost": access_cost,
            "historical_penalty": historical_penalty,
            "overshoot_kw": overshoot_kw,
            "distribution_variable": distribution_variable,
            "odv_cost": odv_cost,
            "surcharges_cost": surcharges_cost,
            "withdrawal_tariff": withdrawal_tariff,
            "total_grid_cost": total_grid,
        }

    # ------------------------------------------------------------------
    # Belastingen
    # ------------------------------------------------------------------

    def calculate_taxes(
        self,
        energy: Dict[str, float],
    ) -> Dict[str, float]:
        c = self.config

        special_excise_cost = (
            energy["p1_volume_mwh"]
            * c.special_excise_eur_mwh
        )

        total_taxes = (
            special_excise_cost
            + c.energy_fund_eur_month
        )

        return {
            "special_excise_cost": special_excise_cost,
            "energy_fund_cost": c.energy_fund_eur_month,
            "total_tax_cost": total_taxes,
        }

    # ------------------------------------------------------------------
    # Invoice rows
    # ------------------------------------------------------------------

    def build_invoice(self) -> InvoiceResult:
        energy = self.calculate_energy()
        grid = self.calculate_grid(energy)
        taxes = self.calculate_taxes(energy)

        c = self.config
        period = self.period_label
        p1_mwh = energy["p1_volume_mwh"]

        rows = []

        def add(
            description: str,
            period_value: str,
            consumption: str,
            unit_price: str,
            total: float,
            vat: str = "",
        ):
            rows.append(
                {
                    "Omschrijving": description,
                    "Periode": period_value,
                    "Verbruik": consumption,
                    "Eenheidsprijs": unit_price,
                    "BTW %": vat,
                    "Totaal excl. BTW": format_euro(total),
                    "_total_numeric": total,
                }
            )

        # ---------------- Energie ----------------
        add(
            "Energiekost",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(
                energy["average_energy_price"], 3
            ),
            energy["energy_cost"],
            f"{format_dutch_number(c.vat_percent, 0)}%",
        )

        add(
            "Bijdrage warmtekracht",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(c.wkc_eur_mwh, 3),
            energy["wkc_cost"],
            f"{format_dutch_number(c.vat_percent, 0)}%",
        )

        add(
            "Korting warmtekracht",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(
                -c.wkc_eur_mwh
                * c.wkc_discount_pct
                / 100.0,
                3,
            ),
            -energy["wkc_discount"],
            f"{format_dutch_number(c.vat_percent, 0)}%",
        )

        add(
            "Bijdrage groene stroom",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(c.gsc_eur_mwh, 3),
            energy["gsc_cost"],
            f"{format_dutch_number(c.vat_percent, 0)}%",
        )

        add(
            "Korting groene stroom",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(
                -c.gsc_eur_mwh
                * c.gsc_discount_pct
                / 100.0,
                3,
            ),
            -energy["gsc_discount"],
            f"{format_dutch_number(c.vat_percent, 0)}%",
        )

        add(
            "Netverlies",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(
                energy["net_loss_price"], 3
            ),
            energy["net_loss_cost"],
            f"{format_dutch_number(c.vat_percent, 0)}%",
        )

        add(
            "Totaal energiekosten",
            "",
            "",
            "",
            energy["total_energy_cost"],
        )

        # ---------------- Nettarieven ----------------
        add(
            "Databeheer",
            f"{self.month}/{self.year}",
            f"{self.days_in_month} dagen",
            f"{format_dutch_number(grid['data_management_day'], 4)} €/dag",
            grid["data_management_month"],
        )

        add(
            "Reactieve energie P1",
            period,
            f"{format_dutch_number(grid['reactive_mvarh'], 3)} MVArh",
            format_unit_price(
                c.reactive_eur_mvarh,
                3,
                "€/MVArh",
            ),
            grid["reactive_cost"],
        )

        add(
            "Capaciteitstarief P1",
            f"{self.month}/{self.year}",
            f"{format_dutch_number(grid['month_peak_kw'], 2)} kW",
            (
                f"{format_dutch_number(grid['capacity_monthly_eur_kw'], 4)} "
                "€/kW/maand"
            ),
            grid["capacity_cost"],
        )

        add(
            "Distributie variabel",
            f"{self.month}/{self.year}",
            f"{format_dutch_number(grid['month_peak_kw'], 2)} kW",
            (
                f"{format_dutch_number(grid['access_power_monthly_eur_kw'], 4)} "
                "€/kW/maand + boete"
            ),
            grid["distribution_variable"],
        )

        add(
            "ODV",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(c.odv_eur_mwh, 3),
            grid["odv_cost"],
        )

        add(
            "Toeslagen",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(c.surcharges_eur_mwh, 3),
            grid["surcharges_cost"],
        )

        add(
            "Afnametarief",
            "",
            "",
            "",
            grid["withdrawal_tariff"],
        )

        add(
            "Totaal nettarieven",
            "",
            "",
            "",
            grid["total_grid_cost"],
        )

        # ---------------- Belastingen ----------------
        add(
            "Bijzondere accijns",
            period,
            f"{format_dutch_number(p1_mwh, 3)} MWh",
            format_unit_price(
                c.special_excise_eur_mwh, 2
            ),
            taxes["special_excise_cost"],
        )

        add(
            "Bijdrage energiefonds",
            f"{self.month}/{self.year}",
            "",
            "",
            taxes["energy_fund_cost"],
        )

        add(
            "Totaal belasting, heffingen en toeslagen",
            "",
            "",
            "",
            taxes["total_tax_cost"],
        )

        subtotal = (
            energy["total_energy_cost"]
            + grid["total_grid_cost"]
            + taxes["total_tax_cost"]
        )

        add(
            "Subtotaal basisbedrag (excl. BTW)",
            "",
            "",
            "",
            subtotal,
        )

        invoice = pd.DataFrame(rows)

        # Interne numerieke kolom niet tonen.
        invoice = invoice[
            [
                "Omschrijving",
                "Periode",
                "Verbruik",
                "Eenheidsprijs",
                "BTW %",
                "Totaal excl. BTW",
            ]
        ]

        metrics = {
            **energy,
            **grid,
            **taxes,
            "subtotal_excl_vat": subtotal,
            "year": float(self.year),
            "month": float(self.month),
        }

        return InvoiceResult(
            invoice=invoice,
            detailed_data=self.data.copy(),
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def bereken_energiefactuur(
    p1: PathLike,
    p2: PathLike,
    hodb: PathLike,
    aerotrim: PathLike,
    kallo: PathLike,
    sapac: PathLike,
    stc: PathLike,
    spotprijzen: PathLike,
    config: Optional[InvoiceConfig] = None,
) -> InvoiceResult:
    """
    Eén functie om de volledige factuur te berekenen.

    Voorbeeld:
        resultaat = bereken_energiefactuur(
            p1="P1.csv",
            p2="P2.csv",
            hodb="HODB.csv",
            aerotrim="Aerotrim.csv",
            kallo="Kallo.csv",
            sapac="SAPAC.csv",
            stc="STC (1).csv",
            spotprijzen="Spotprijs.csv",
        )

        print(resultaat)
        resultaat.invoice.to_excel("factuur.xlsx", index=False)
    """
    if config is None:
        config = InvoiceConfig()

    meter_paths = {
        "P1": p1,
        "P2": p2,
        "HODB": hodb,
        "Aerotrim": aerotrim,
        "Kallo": kallo,
        "SAPAC": sapac,
        "STC": stc,
    }

    reader = MeterDataReader()

    combined = reader.combine_meters(meter_paths)
    spot = reader.load_spot_prices(spotprijzen)
    combined = reader.attach_spot_prices(combined, spot)

    # P1 wordt afzonderlijk ingelezen omdat naast actief volume ook
    # 'Afname Reactief' nodig is.
    p1_full = reader.read_meter_csv(p1, meter_name="P1")

    calculator = VlaamseEnergiefactuurCalculator(
        config=config,
        data=combined,
        p1_full_data=p1_full,
    )

    return calculator.build_invoice()


# ---------------------------------------------------------------------------
# Command line / direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Pas alleen de paden hieronder aan.

    De bestanden hoeven niet in dezelfde map te staan als deze module.
    Gebruik bijvoorbeeld:
        data/P1.csv
    of een absoluut Windows-pad.
    """

    BASE = Path(__file__).resolve().parent
    DATA = BASE / "data"

    # ------------------------------------------------------------------
    # BESTANDEN
    # ------------------------------------------------------------------
    #
    # Verwachte bestanden:
    #
    #   data/
    #       P1.csv
    #       P2.csv
    #       HODB.csv
    #       Aerotrim.csv
    #       Kallo.csv
    #       SAPAC.csv
    #       STC (1).csv
    #       Spotprijs.csv
    #
    # ------------------------------------------------------------------

    files = {
        "p1": DATA / "P1.csv",
        "p2": DATA / "P2.csv",
        "hodb": DATA / "HODB.csv",
        "aerotrim": DATA / "Aerotrim.csv",
        "kallo": DATA / "Kallo.csv",
        "sapac": DATA / "SAPAC.csv",
        "stc": DATA / "STC (1).csv",
        "spotprijzen": DATA / "Spotprijs.csv",
    }

    missing = [
        str(path)
        for path in files.values()
        if not Path(path).exists()
    ]

    if missing:
        print()
        print("BESTANDEN ONTBREKEN:")
        for path in missing:
            print(f"  - {path}")
        print()
        print(
            "Plaats de CSV-bestanden in de map 'data' naast deze "
            "Python-module, of pas DATA hierboven aan."
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # CONFIGURATIE
    # ------------------------------------------------------------------
    config = InvoiceConfig(
        forward_volume_mw=1.3,
        forward_price_eur_mwh=85.58,
        onbalans_eur_mwh=4.612,
        go_eur_mwh=3.04,
        wkc_eur_mwh=3.556,
        gsc_eur_mwh=10.835,
        wkc_discount_pct=0.47,
        gsc_discount_pct=0.47,
        net_loss_pct=1.85,
        capacity_annual_eur_kw=46.8630636,
        data_management_annual_eur=57.65,
        reactive_eur_mvarh=13.4149,
        access_power_annual_eur_kw=31.5677340,
        access_power_p1_kw=2850.0,
        historical_penalty_eur_kw=71.0,
        overshoot_multiplier=1.5,
        odv_eur_mwh=4.0283,
        surcharges_eur_mwh=0.1922,
        special_excise_eur_mwh=10.69,
        energy_fund_eur_month=189.48,
        vat_percent=21.0,
    )

    result = bereken_energiefactuur(
        **files,
        config=config,
    )

    print()
    print(result)
    print()
    print("DETAIL FACTUUR")
    print("=" * 92)
    print(result.invoice.to_string(index=False))
    print()

    # Optioneel:
    # result.to_excel(BASE / "energiefactuur.xlsx")
