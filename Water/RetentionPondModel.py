import os
import numpy as np
import math
import pandas as pd
import fastparquet
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.optimize import fsolve
import time as t
import warnings
from numba import njit, prange
import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

@njit
def volume_function_jit(d):
    # V = π (15.21 d + 11.7 d^2 + 3 d^3)
    return math.pi * (15.21 * d + 11.7 * d * d + 3.0 * d * d * d)


def build_volume_depth_table(max_depth, n_points=100001):
    max_vol = math.pi * (15.21 * max_depth + 11.7 * max_depth**2 + 3.0 * max_depth**3)
    V_vals = np.linspace(0.0, max_vol, n_points)
    d_vals = np.zeros(n_points, dtype=np.float64)
    for i, V in enumerate(V_vals):
        # Solve analytic cubic via fsolve outside of JIT region
        d_root = fsolve(lambda x: math.pi*(15.21*x + 11.7*x**2 + 3*x**3) - V,
                        x0=max_depth/2)[0]
        d_vals[i] = d_root
    return V_vals, d_vals

@njit
def preissmann_slot_outflow_jit(prev_V, prev_D, pipe_d, slot_w, S0, n):
    d_eff = prev_D - 1
    if d_eff <= 0.0:
        return 0.0
    if d_eff <= pipe_d:
        r = pipe_d * 0.5
        theta = 2.0 * math.acos((r - d_eff) / r)
        A = (pipe_d * pipe_d / 8.0) * (theta - math.sin(theta))
        P = r * theta
        return (A**(5.0/3.0)) * math.sqrt(S0) / (n * (P**(2.0/3.0)))
    else:
        A_open = math.pi * (pipe_d/2.0)**2
        A_slot = slot_w * (d_eff - pipe_d)
        A_tot = A_open + A_slot
        P_tot = math.pi * pipe_d + 2.0*(d_eff - pipe_d)
        R = A_tot / P_tot
        return (1.0/n) * A_tot * (R**(2.0/3.0)) * math.sqrt(S0)

@njit
def evap_rate_jit(prev_V, V_vals, d_vals):
    # Interpolate depth
    idx = np.searchsorted(V_vals, prev_V)
    if idx == 0:
        depth = d_vals[0]
    elif idx >= V_vals.shape[0]:
        depth = d_vals[-1]
    else:
        x0, x1 = V_vals[idx-1], V_vals[idx]
        y0, y1 = d_vals[idx-1], d_vals[idx]
        depth = y0 + (prev_V - x0)*(y1-y0)/(x1-x0)
    area = math.pi * (3.9 + 3.0 * depth)**2
    return 0.035 * area / (7.0*24.0*3600.0)

@njit
def simulate_step_jit(inflow, V_vals, d_vals, slot_w, pipe_d, S0, n, last_V, last_D, max_vol):
    n_steps = inflow.shape[0]

    Vol = np.empty(n_steps, dtype=np.float64)
    Dep = np.empty(n_steps, dtype=np.float64)
    Qout = np.empty(n_steps, dtype=np.float64)
    Ev = np.empty(n_steps, dtype=np.float64)
    Ov = np.empty(n_steps, dtype=np.float64)

    prev_V = last_V
    prev_D = last_D

    for i in range(n_steps):
        # 1) Evapotranspiration
        Ev[i] = evap_rate_jit(prev_V, V_vals, d_vals)

        # 2) Outflow
        Qout[i] = preissmann_slot_outflow_jit(prev_V, prev_D, pipe_d, slot_w, S0, n)

        # 3) Volume update & overflow
        newV = prev_V + inflow[i] - Qout[i] - Ev[i]
        if newV <= 0.0:
            Vol[i], Ov[i] = 0.0, 0.0
        elif newV >= max_vol:
            Vol[i], Ov[i] = max_vol, newV - max_vol
        else:
            Vol[i], Ov[i] = newV, 0.0

        # 4) Depth via lookup
        idx = np.searchsorted(V_vals, Vol[i])
        if idx == 0:
            Dep[i] = d_vals[0]
        elif idx >= V_vals.shape[0]:
            Dep[i] = d_vals[-1]
        else:
            x0, x1 = V_vals[idx-1], V_vals[idx]
            y0, y1 = d_vals[idx-1], d_vals[idx]
            Dep[i] = y0 + (Vol[i] - x0) * (y1 - y0) / (x1 - x0)

        # 5) Prepare for next step
        prev_V, prev_D = Vol[i], Dep[i]

    return Vol, Dep, Qout, Ev, Ov

def create_runoff_volume(df):
    df["Inflow"] = df["Amount"] * 83900 * 0.72 / 1000
    return df


def hyetograph_function(time):
    return (time * ((1 - time)**4)) / 0.08192


def create_second_dt_df(clipped):
    if not np.issubdtype(clipped.index.dtype, np.datetime64):
        clipped.index = pd.to_datetime(clipped.index, format='%d/%m/%Y')
    frames = []
    t_global = np.linspace(0,1,(6)*3600,endpoint=False)
    hyeto_norm = hyetograph_function(t_global)
    hyeto_norm /= hyeto_norm.sum()
    for day, row in clipped.iterrows():
        day_start = pd.Timestamp(day).normalize()
        full_idx = pd.date_range(day_start, day_start+pd.Timedelta(days=1)-pd.Timedelta(seconds=1), freq='s')
        rainfall = np.zeros(len(full_idx))
        dist = hyeto_norm * row['Amount']
        rainfall[:len(dist)] = dist
        frames.append(pd.DataFrame({'Amount': rainfall}, index=full_idx))
    return pd.concat(frames)

def run_hydrailic_model(model, max_volume, slot_width, pipe_d, S0, n, last_V, last_D):
    inflow = model['Inflow'].to_numpy()
    Vol, Dep, Qout, Ev, Ov = simulate_step_jit(
        inflow, V_lookup, d_lookup,
        slot_width, pipe_d, S0, n,
        last_V, last_D, max_volume
    )
    model['Pond Volume']       = Vol
    model['Depth']             = Dep
    model['Outflow']           = Qout
    model['Evapotranspiration'] = Ev
    model['Overflow']          = Ov
    return model

if __name__ == "__main__":
    max_depth = 1.2
    max_volume = math.pi * (15.21 * max_depth + 11.7 * max_depth**2 + 3*max_depth**3)
    print(f"Max volume: {max_volume} m3")
    V_lookup, d_lookup = build_volume_depth_table(max_depth)

    # Physical properties
    depth_to_drain = 11.5
    distance_to_drain = 300
    pipe_diameter = 0.20
    S0 = depth_to_drain / distance_to_drain
    ks = 0.0015e-3
    n_coeff = 0.038 * (ks**(1/6))

    cwd = os.getcwd()
    print(cwd)
    file_path = os.path.join(cwd, "Final Rainfall Datasets", "WETTERtimeseriescopy.csv")
    daily_series = pd.read_csv(file_path, index_col="Date")

    last_volume, last_depth = 0.0, 0.0
    save_dir = os.path.join(cwd, "Pond Simulation", "ParquetFilesAEP")
    os.makedirs(save_dir, exist_ok=True)

    for snip in tqdm.tqdm(range(0, len(daily_series)//15)):
        snippet = daily_series[snip*15:(snip*15)+15]
        date = snippet.index[0]
        file_name = f"{date[0:2]}-{date[3:5]}-{date[6:10]}"

        sec_df = create_second_dt_df(snippet)
        sec_df = create_runoff_volume(sec_df)
        sec_df = run_hydrailic_model(sec_df, max_volume, 0.01, pipe_diameter, S0, n_coeff, last_volume, last_depth)

        sec_df.to_parquet(os.path.join(save_dir, f"{file_name}.parquet"), engine="fastparquet")

        last_row = sec_df.iloc[-1]
        last_volume, last_depth = last_row['Pond Volume'], last_row['Depth']