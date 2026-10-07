"""
Cognitive CanSat - Automated Post-Flight Analysis & Mission Reporting System
Module: ml/flight_analyzer.py
Author: Rishi Nayak (Data Analytics & Flight Dynamics Lead)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use('Agg')  # Headless rendering for automated reports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Automatically resolve canonical project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "test_cases")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "reports")


class FlightAnalyzer:
    """
    Ingests and analyzes sounding pico-satellite telemetry records.
    Produces publication-grade figures, kinematics statistics, and PDF reports.
    """

    def __init__(self, csv_path: str, output_dir: str = None):
        if not os.path.isabs(csv_path):
            if not os.path.exists(csv_path) and os.path.exists(os.path.join(BASE_DIR, csv_path)):
                csv_path = os.path.join(BASE_DIR, csv_path)

        self.csv_path = os.path.abspath(csv_path)
        
        if output_dir is None:
            self.output_dir = DEFAULT_OUTPUT_DIR
        elif not os.path.isabs(output_dir):
            self.output_dir = os.path.join(BASE_DIR, output_dir)
        else:
            self.output_dir = output_dir

        self.df = None
        self.stats = {}

        self.pdf_dir = os.path.join(self.output_dir, "pdf")
        self.fig_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        self.mission_name = os.path.splitext(os.path.basename(self.csv_path))[0]

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads CSV and maps column naming conventions across flight test cases."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Telemetry CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # 1. Temporal Normalization (Handles ISO-8601 timestamps or numeric time)
        if 'time' in self.df.columns and np.issubdtype(self.df['time'].dtype, np.number):
            self.df['time'] = self.df['time'] - self.df['time'].iloc[0]
        elif 'timestamp' in self.df.columns:
            t_parsed = pd.to_datetime(self.df['timestamp'], errors='coerce')
            if t_parsed.notna().all():
                dt_sec = (t_parsed - t_parsed.iloc[0]).dt.total_seconds()
                self.df['time'] = dt_sec.values
            else:
                self.df['time'] = np.arange(len(self.df)) * 0.8
        else:
            self.df['time'] = np.arange(len(self.df)) * 0.8

        # 2. Forward/Back Fill Sensor Dropouts
        self.df = self.df.bfill().ffill()

        # 3. Canonical Column Normalization
        if 'altitude' not in self.df.columns:
            for c in ['Altitude_m', 'alt', 'altitude_m']:
                if c in self.df.columns: self.df['altitude'] = self.df[c]; break

        if 'pressure' not in self.df.columns:
            for c in ['Pressure_Pa', 'press']:
                if c in self.df.columns:
                    self.df['pressure'] = self.df[c] / 100.0 if self.df[c].median() > 5000 else self.df[c]
                    break

        if 'temperature' not in self.df.columns:
            for c in ['temp', 'Temp_C']:
                if c in self.df.columns: self.df['temperature'] = self.df[c]; break

        if 'accel_mag' not in self.df.columns:
            if 'accelMag' in self.df.columns:
                self.df['accel_mag'] = self.df['accelMag']
            elif {'ax', 'ay', 'az'}.issubset(self.df.columns):
                self.df['accel_mag'] = np.sqrt(self.df['ax']**2 + self.df['ay']**2 + self.df['az']**2)
            elif {'accel_x', 'accel_y', 'accel_z'}.issubset(self.df.columns):
                self.df['accel_mag'] = np.sqrt(self.df['accel_x']**2 + self.df['accel_y']**2 + self.df['accel_z']**2)
            else:
                self.df['accel_mag'] = 1.0

        # 4. Altitude Savitzky-Golay Filtering (Noise Reduction)
        n_samples = len(self.df)
        if n_samples >= 7:
            window_length = min(11, n_samples // 2 * 2 + 1)
            poly_order = min(3, window_length - 2)
            self.df['alt_smoothed'] = savgol_filter(self.df['altitude'].values, window_length, poly_order)
        else:
            self.df['alt_smoothed'] = self.df['altitude'].values

        # 5. Vertical Velocity (dz/dt)
        self.df['velocity'] = np.gradient(self.df['alt_smoothed'].values, self.df['time'].values)

        return self.df

    def calculate_statistics(self) -> dict:
        """Extracts key mission flight dynamics and atmospheric indicators."""
        apogee_idx = self.df['alt_smoothed'].idxmax()

        # Touchdown shock in final 15% of flight
        late_flight_start = int(len(self.df) * 0.85)
        late_flight_accel = self.df['accel_mag'].iloc[late_flight_start:]
        touchdown_shock = late_flight_accel.max() if len(late_flight_accel) > 0 else self.df['accel_mag'].iloc[-1]

        # Environmental Lapse Rate (°C / 100m)
        alt_range = self.df['alt_smoothed'].max() - self.df['alt_smoothed'].min()
        temp_range = self.df['temperature'].max() - self.df['temperature'].min()
        elr = round((temp_range / max(alt_range, 10.0)) * 100.0, 3)

        self.stats = {
            'Mission Name': self.mission_name,
            'Apogee (m)': round(float(self.df['alt_smoothed'].max()), 2),
            'Time to Apogee (s)': round(float(self.df['time'].iloc[apogee_idx]), 2),
            'Max Ascent Velocity (m/s)': round(float(self.df['velocity'].max()), 2),
            'Max Descent Velocity (m/s)': round(float(self.df['velocity'].min()), 2),
            'Max G-Shock (Gs)': round(float(self.df['accel_mag'].max()), 2),
            'Touchdown Shock (Gs)': round(float(touchdown_shock), 2),
            'Mission Duration (s)': round(float(self.df['time'].max()), 2),
            'Min Temperature (°C)': round(float(self.df['temperature'].min()), 2),
            'Max Temperature (°C)': round(float(self.df['temperature'].max()), 2),
            'Min Pressure (hPa)': round(float(self.df['pressure'].min()), 2),
            'Max Pressure (hPa)': round(float(self.df['pressure'].max()), 2),
            'Lapse Rate (°C/100m)': elr,
            'Telemetry Frames': int(len(self.df))
        }

        for lat_col in ['lat', 'latitude', 'GPS_Lat']:
            if lat_col in self.df.columns:
                self.stats['Landing Lat'] = round(float(self.df[lat_col].iloc[-1]), 6); break
        for lon_col in ['lon', 'longitude', 'GPS_Lon']:
            if lon_col in self.df.columns:
                self.stats['Landing Lon'] = round(float(self.df[lon_col].iloc[-1]), 6); break

        return self.stats

    def generate_plots_and_report(self):
        """Generates individual 300 DPI PNGs, a 4-panel overview, and a combined PDF report."""
        pdf_path = os.path.join(self.pdf_dir, f"{self.mission_name}_Flight_Report.pdf")

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'axes.grid': True,
            'grid.alpha': 0.5,
            'grid.linestyle': '--',
            'figure.autolayout': True
        })

        with PdfPages(pdf_path) as pdf:
            # 1. Summary Table
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
            ax.axis('tight'); ax.axis('off')
            table_data = [[k, str(v)] for k, v in self.stats.items()]
            table = ax.table(cellText=table_data, colLabels=["Flight Dynamics Parameter", "Calibrated Value"], loc='center', cellLoc='center')
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 1.4)
            for i in range(2):
                table[(0, i)].set_facecolor('#1f3b5c')
                table[(0, i)].set_text_props(color='white', weight='bold')
            for row in range(1, len(table_data) + 1):
                bg_color = '#f2f6fa' if row % 2 == 0 else '#ffffff'
                for col in range(2): table[(row, col)].set_facecolor(bg_color)
            plt.title(f"Cognitive CanSat - Mission Summary: {self.mission_name}", fontsize=14, fontweight='bold', pad=20, color='#0f2038')
            pdf.savefig(fig, bbox_inches='tight')
            fig.savefig(os.path.join(self.fig_dir, f"{self.mission_name}_Summary_Table.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

            # 2. Altitude vs Time
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
            ax.plot(self.df['time'], self.df['altitude'], color='#90b4ce', alpha=0.6, label='Raw Barometric')
            ax.plot(self.df['time'], self.df['alt_smoothed'], color='#1d3557', lw=2.2, label='Smoothed')
            apogee_t = self.stats['Time to Apogee (s)']
            apogee_alt = self.stats['Apogee (m)']
            ax.axvline(x=apogee_t, color='#e63946', linestyle='--', lw=1.5, label=f"Apogee ({apogee_alt:.1f} m)")
            ax.scatter([apogee_t], [apogee_alt], color='#e63946', s=50, zorder=5)
            ax.set_title(f"Altitude vs. Time — {self.mission_name}", fontweight='bold', color='#1d3557')
            ax.set_xlabel("Mission Elapsed Time (s)"); ax.set_ylabel("Altitude Above Launch Pad (m)")
            ax.legend(loc='upper right', framealpha=0.9)
            self._save_fig(fig, pdf, "Altitude_vs_Time")

            # 3. Velocity Profile
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
            ax.plot(self.df['time'], self.df['velocity'], color='#2a9d8f', lw=2.0, label='Vertical Velocity (dz/dt)')
            ax.axhline(0, color='#1d3557', linestyle='-', lw=1.0)
            ax.axhspan(-6.0, -4.0, color='#52b788', alpha=0.25, label='Canopy (-4 to -6 m/s)')
            ax.axvline(x=apogee_t, color='#e63946', linestyle=':', lw=1.2, label="Apogee Transition")
            ax.set_title(f"Vertical Velocity Profile — {self.mission_name}", fontweight='bold', color='#1d3557')
            ax.set_xlabel("Mission Elapsed Time (s)"); ax.set_ylabel("Vertical Velocity (m/s)")
            ax.legend(loc='upper right', framealpha=0.9)
            self._save_fig(fig, pdf, "Velocity_Profile")

            # 4. G-Shock Profile
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
            ax.plot(self.df['time'], self.df['accel_mag'], color='#7209b7', lw=1.5, label='Resultant Acceleration (|G|)')
            ax.axhline(1.0, color='#6c757d', linestyle=':', lw=1.0, label='1.0 G Gravity')
            max_g = self.stats['Max G-Shock (Gs)']
            max_g_time = self.df.loc[self.df['accel_mag'].idxmax(), 'time']
            ax.scatter([max_g_time], [max_g], color='#d90429', s=60, zorder=5)
            ax.annotate(f"Peak: {max_g:.2f} G\n(T+{max_g_time:.1f}s)", xy=(max_g_time, max_g),
                        xytext=(max_g_time + max(2.0, self.df['time'].max() * 0.05), max_g * 0.9),
                        arrowprops=dict(facecolor='#d90429', shrink=0.08, width=1, headwidth=6),
                        fontweight='bold', color='#d90429')
            ax.set_title(f"G-Shock & Acceleration — {self.mission_name}", fontweight='bold', color='#1d3557')
            ax.set_xlabel("Mission Elapsed Time (s)"); ax.set_ylabel("Acceleration Magnitude (Gs)")
            ax.legend(loc='upper right', framealpha=0.9)
            self._save_fig(fig, pdf, "G_Shock_Profile")

            # 5. Sounding Profile
            fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
            ax2 = ax1.twiny()
            l1 = ax1.plot(self.df['pressure'], self.df['alt_smoothed'], color='#0077b6', lw=2.0, label='Pressure (hPa)')
            l2 = ax2.plot(self.df['temperature'], self.df['alt_smoothed'], color='#e76f51', lw=2.0, label='Temperature (°C)')
            ax1.set_xlabel("Barometric Pressure (hPa)", color='#0077b6', fontweight='bold')
            ax2.set_xlabel("Ambient Temperature (°C)", color='#e76f51', fontweight='bold')
            ax1.set_ylabel("Altitude Above Pad (m)", fontweight='bold')
            ax1.tick_params(axis='x', labelcolor='#0077b6'); ax2.tick_params(axis='x', labelcolor='#e76f51')
            ax1.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='lower left', framealpha=0.9)
            plt.title(f"Atmospheric Sounding (Lapse Rate: {self.stats['Lapse Rate (°C/100m)']} °C/100m)", pad=20, fontweight='bold', color='#1d3557')
            self._save_fig(fig, pdf, "Atmospheric_Sounding")

            # 6. Consolidated 4-Panel Figure
            fig, axs = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
            fig.suptitle(f"Cognitive CanSat Mission Dynamics: {self.mission_name}", fontsize=15, fontweight='bold', color='#0f2038', y=0.98)
            axs[0, 0].plot(self.df['time'], self.df['altitude'], color='#90b4ce', alpha=0.5)
            axs[0, 0].plot(self.df['time'], self.df['alt_smoothed'], color='#1d3557', lw=1.8)
            axs[0, 0].set_title("A. Trajectory Profile", fontweight='bold')
            axs[0, 0].set_xlabel("Time (s)"); axs[0, 0].set_ylabel("Altitude (m)")

            axs[0, 1].plot(self.df['time'], self.df['velocity'], color='#2a9d8f', lw=1.8)
            axs[0, 1].axhline(0, color='#1d3557', linestyle='-', lw=0.8)
            axs[0, 1].set_title("B. Vertical Velocity Curve", fontweight='bold')
            axs[0, 1].set_xlabel("Time (s)"); axs[0, 1].set_ylabel("Velocity (m/s)")

            axs[1, 0].plot(self.df['time'], self.df['accel_mag'], color='#7209b7', lw=1.3)
            axs[1, 0].set_title(f"C. G-Shock (Peak: {max_g:.2f}G)", fontweight='bold')
            axs[1, 0].set_xlabel("Time (s)"); axs[1, 0].set_ylabel("Acceleration (Gs)")

            ax1_sub = axs[1, 1]; ax2_sub = ax1_sub.twiny()
            ax1_sub.plot(self.df['pressure'], self.df['alt_smoothed'], color='#0077b6', lw=1.5)
            ax2_sub.plot(self.df['temperature'], self.df['alt_smoothed'], color='#e76f51', lw=1.5)
            axs[1, 1].set_title("D. Atmospheric Sounding Profile", fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self._save_fig(fig, pdf, "Publication_4Panel")

        print(f"[OK] Publication Report generated: {pdf_path}")
        return pdf_path

    def _save_fig(self, fig, pdf, filename_suffix: str):
        pdf.savefig(fig, bbox_inches='tight')
        png_path = os.path.join(self.fig_dir, f"{self.mission_name}_{filename_suffix}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def analyze_single_flight(csv_path: str, output_dir: str = None) -> dict:
    if output_dir is None: output_dir = DEFAULT_OUTPUT_DIR
    analyzer = FlightAnalyzer(csv_path, output_dir)
    analyzer.load_and_preprocess()
    stats = analyzer.calculate_statistics()
    analyzer.generate_plots_and_report()
    return stats


def analyze_all_test_cases(data_dir: str = None, output_dir: str = None) -> pd.DataFrame:
    if data_dir is None: data_dir = DEFAULT_DATA_DIR
    elif not os.path.isabs(data_dir) and not os.path.exists(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)

    if output_dir is None: output_dir = DEFAULT_OUTPUT_DIR
    elif not os.path.isabs(output_dir): output_dir = os.path.join(BASE_DIR, output_dir)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files: raise FileNotFoundError(f"No CSVs found in: {data_dir}")

    print(f"\n[i] Initiating Batch Flight Analysis for {len(csv_files)} CanSat scenarios in '{data_dir}'...")
    all_stats = [analyze_single_flight(f, output_dir) for f in csv_files]

    summary_df = pd.DataFrame(all_stats)
    summary_csv = os.path.join(output_dir, "all_missions_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n[OK] Batch analysis complete! Master summary saved to: {summary_csv}")
    cols = ['Mission Name', 'Apogee (m)', 'Time to Apogee (s)', 'Max Descent Velocity (m/s)', 'Max G-Shock (Gs)', 'Touchdown Shock (Gs)', 'Mission Duration (s)']
    print("\n" + "="*95)
    print("                      COGNITIVE CANSAT FLIGHT BENCHMARK MATRIX")
    print("="*95)
    print(summary_df[[c for c in cols if c in summary_df.columns]].to_string(index=False))
    print("="*95 + "\n")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive CanSat Automated Post-Flight Telemetry Analyzer")
    parser.add_argument("--file", help="Path to single CSV (e.g. test_cases/01_nominal_sounding_flight.csv)")
    parser.add_argument("--dir", default=DEFAULT_DATA_DIR, help="Directory containing CSV test cases")
    parser.add_argument("--all", action="store_true", help="Batch analyze all CSV files in --dir")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for generated PDFs and PNGs")

    args = parser.parse_args()

    if args.file:
        analyze_single_flight(args.file, args.output)
    else:
        analyze_all_test_cases(args.dir, args.output)