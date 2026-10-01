import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_mobility_data(input_file="data/interim/mobility_parsed.csv", output_dir="data/processed", results_dir="results"):
    input_path = Path(input_file)
    output_dir_path = Path(output_dir)
    results_tables = Path(results_dir) / "tables"
    results_figures = Path(results_dir) / "figures"
    
    output_dir_path.mkdir(parents=True, exist_ok=True)
    results_tables.mkdir(parents=True, exist_ok=True)
    results_figures.mkdir(parents=True, exist_ok=True)
    
    clean_output_file = output_dir_path / "mobility_clean.csv"
    vehicle_summary_file = output_dir_path / "vehicle_summary.csv"
    data_quality_report_file = results_tables / "data_quality_report.csv"
    preprocessing_log_file = results_tables / "preprocessing_log.csv"
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist. Please complete Phase 1 first.")
        sys.exit(1)
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    if df.empty:
        print("Error: Input file is empty.")
        sys.exit(1)
        
    original_records = len(df)
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    print("\n--- STEP 1: LOAD DATA ---")
    print(f"File Path: {input_path}")
    print(f"Number of Rows: {original_records}")
    print(f"Number of Columns: {df.shape[1]}")
    print(f"Column Names: {list(df.columns)}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Memory Usage: {original_memory:.2f} MB")
    
    print("\n--- STEP 2: DATASET STRUCTURE INSPECTION ---")
    print(f"Shape: {df.shape}")
    print("\nHead:\n", df.head())
    print("\nTail:\n", df.tail())
    print("\nInfo:")
    df.info()
    print("\nDescribe:\n", df.describe(include='all'))
    
    print("\n--- STEP 3: MISSING VALUE ANALYSIS ---")
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / original_records) * 100
    for col in df.columns:
        print(f"Column '{col}': {missing_counts[col]} missing ({missing_pct[col]:.4f}%)")
        
    preprocessing_logs = []
    
    # Missing Value Policy Execution
    print("\n--- MISSING VALUE HANDLING ---")
    missing_time = df['time'].isna().sum()
    missing_vid = df['vehicle_id'].isna().sum()
    missing_x = df['x'].isna().sum()
    missing_y = df['y'].isna().sum()
    missing_speed = df['speed'].isna().sum() if 'speed' in df.columns else 0
    
    # Drop rows with missing time, vehicle_id, x, y
    records_before = len(df)
    df = df.dropna(subset=['time', 'vehicle_id', 'x', 'y'])
    records_after = len(df)
    if records_before != records_after:
        preprocessing_logs.append({
            'operation': 'Drop missing core fields',
            'records_before': records_before,
            'records_after': records_after,
            'records_affected': records_before - records_after,
            'reason': 'Missing time, vehicle_id, x, or y which are mandatory for spatial/temporal analysis.'
        })
        print(f"Dropped {records_before - records_after} records due to missing core fields.")
    
    if 'speed' in df.columns and missing_speed > 0:
        preprocessing_logs.append({
            'operation': 'Retain missing speed',
            'records_before': len(df),
            'records_after': len(df),
            'records_affected': missing_speed,
            'reason': 'Speed left as NaN where unavailable; no arbitrary fabrication.'
        })
        print(f"Retained {missing_speed} records with missing speed.")
        
    print("\n--- STEP 4: DATA TYPE VALIDATION ---")
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    if 'speed' in df.columns:
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        
    invalid_time = df['time'].isna().sum()
    invalid_x = df['x'].isna().sum()
    invalid_y = df['y'].isna().sum()
    if invalid_time > 0 or invalid_x > 0 or invalid_y > 0:
        records_before = len(df)
        df = df.dropna(subset=['time', 'x', 'y'])
        records_after = len(df)
        preprocessing_logs.append({
            'operation': 'Drop coerced NaNs',
            'records_before': records_before,
            'records_after': records_after,
            'records_affected': records_before - records_after,
            'reason': 'Invalid numeric values coerced to NaN in core fields.'
        })
        
    df['vehicle_id'] = df['vehicle_id'].astype(int)
    
    print("\n--- STEP 5: DUPLICATE ANALYSIS ---")
    total_duplicates = df.duplicated().sum()
    print(f"Total exact duplicate rows: {total_duplicates}")
    mobility_duplicates = df.duplicated(subset=['time', 'vehicle_id']).sum()
    print(f"Total mobility duplicates (time, vehicle_id): {mobility_duplicates}")
    
    if total_duplicates > 0:
        records_before = len(df)
        df = df.drop_duplicates()
        records_after = len(df)
        preprocessing_logs.append({
            'operation': 'Remove exact duplicates',
            'records_before': records_before,
            'records_after': records_after,
            'records_affected': records_before - records_after,
            'reason': 'Exact duplicate rows provide no informational value.'
        })
        print(f"Removed {records_before - records_after} exact duplicate rows.")
        
    print("\n--- STEP 6: TEMPORAL CONSISTENCY ---")
    min_time = df['time'].min()
    max_time = df['time'].max()
    unique_timestamps = df['time'].nunique()
    print(f"Min time: {min_time}, Max time: {max_time}, Unique timestamps: {unique_timestamps}")
    
    sorted_times = np.sort(df['time'].unique())
    time_diffs = np.diff(sorted_times)
    if len(time_diffs) > 0:
        min_dt = np.min(time_diffs)
        max_dt = np.max(time_diffs)
        median_dt = np.median(time_diffs)
        mode_dt = pd.Series(time_diffs).mode()[0]
        print(f"Min dt: {min_dt}, Max dt: {max_dt}, Median dt: {median_dt}, Mode dt: {mode_dt}")
        if min_dt == max_dt:
            print("Time steps are constant.")
        else:
            print("Time steps are irregular.")
            
    print("\n--- STEP 7: VEHICLE TEMPORAL CONSISTENCY ---")
    vehicle_summary = df.groupby('vehicle_id').agg(
        first_time=('time', 'min'),
        last_time=('time', 'max'),
        record_count=('time', 'count'),
        unique_timestamps=('time', 'nunique')
    ).reset_index()
    
    if 'speed' in df.columns:
        speed_stats = df.groupby('vehicle_id')['speed'].agg(
            mean_speed='mean',
            min_speed='min',
            max_speed='max'
        ).reset_index()
        vehicle_summary = vehicle_summary.merge(speed_stats, on='vehicle_id')
        
    vehicle_summary.to_csv(vehicle_summary_file, index=False)
    print(f"Saved vehicle summary to {vehicle_summary_file}")
    
    print("\n--- STEP 8: SPATIAL DATA VALIDATION ---")
    min_x = df['x'].min()
    max_x = df['x'].max()
    min_y = df['y'].min()
    max_y = df['y'].max()
    print(f"X range: {min_x} to {max_x}")
    print(f"Y range: {min_y} to {max_y}")
    
    print("\n--- STEP 9: SPEED VALIDATION ---")
    if 'speed' in df.columns:
        min_speed = df['speed'].min()
        max_speed = df['speed'].max()
        print(f"Speed range: {min_speed} to {max_speed}")
        print(f"Speed mean: {df['speed'].mean():.2f}, Speed median: {df['speed'].median():.2f}")
        
    print("\n--- STEP 10: OUTLIER ANALYSIS ---")
    for col in ['x', 'y', 'speed']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            pct_outliers = (len(outliers) / len(df)) * 100
            print(f"Potential outliers in {col}: {len(outliers)} ({pct_outliers:.2f}%)")
            if len(outliers) > 0:
                preprocessing_logs.append({
                    'operation': f'Outlier analysis for {col}',
                    'records_before': len(df),
                    'records_after': len(df),
                    'records_affected': len(outliers),
                    'reason': 'Outliers identified using IQR but retained as valid simulation data.'
                })
                
    print("\n--- STEP 11 & 12: VEHICLE TRAJECTORY CONSISTENCY ---")
    df_sorted = df.sort_values(by=['vehicle_id', 'time']).copy()
    df_sorted['dt'] = df_sorted.groupby('vehicle_id')['time'].diff()
    df_sorted['dx'] = df_sorted.groupby('vehicle_id')['x'].diff()
    df_sorted['dy'] = df_sorted.groupby('vehicle_id')['y'].diff()
    df_sorted['distance'] = np.sqrt(df_sorted['dx']**2 + df_sorted['dy']**2)
    df_sorted['derived_speed'] = df_sorted['distance'] / df_sorted['dt']
    
    suspicious = df_sorted[df_sorted['dt'] <= 0]
    print(f"Suspicious dt <= 0: {len(suspicious)} records")
    
    print("\n--- STEP 13: SORTING ---")
    df_clean = df.sort_values(by=['time', 'vehicle_id']).copy()
    
    if 'source_line' in df_clean.columns:
        df_clean = df_clean.drop(columns=['source_line'])
        
    print("\n--- STEP 14: CLEAN DATASET ---")
    df_clean.to_csv(clean_output_file, index=False)
    print(f"Cleaned dataset saved to {clean_output_file}")
    
    print("\n--- STEP 16: DATA QUALITY REPORT ---")
    dq_report = {
        'original_records': original_records,
        'cleaned_records': len(df_clean),
        'removed_records': original_records - len(df_clean),
        'duplicate_records': total_duplicates,
        'missing_time_records': missing_time,
        'missing_vehicle_records': missing_vid,
        'missing_x_records': missing_x,
        'missing_y_records': missing_y,
        'missing_speed_records': missing_speed,
        'unique_vehicles': df_clean['vehicle_id'].nunique(),
        'unique_timestamps': df_clean['time'].nunique(),
        'min_time': min_time,
        'max_time': max_time,
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'min_speed': min_speed if 'speed' in df.columns else np.nan,
        'max_speed': max_speed if 'speed' in df.columns else np.nan
    }
    
    dq_df = pd.DataFrame(list(dq_report.items()), columns=['metric', 'value'])
    dq_df.to_csv(data_quality_report_file, index=False)
    print(f"Data quality report saved to {data_quality_report_file}")
    
    print("\n--- STEP 18: PREPROCESSING LOG ---")
    if preprocessing_logs:
        log_df = pd.DataFrame(preprocessing_logs)
    else:
        log_df = pd.DataFrame(columns=['operation', 'records_before', 'records_after', 'records_affected', 'reason'])
        
    log_df.loc[len(log_df)] = {
        'operation': 'Save clean dataset',
        'records_before': len(df_clean),
        'records_after': len(df_clean),
        'records_affected': 0,
        'reason': 'Saved final sorted, unnormalized dataset with source_line removed.'
    }
    log_df.to_csv(preprocessing_log_file, index=False)
    print(f"Preprocessing log saved to {preprocessing_log_file}")
    
    print("\n--- STEP 19: VISUALIZATIONS ---")
    plt.figure(figsize=(10, 10))
    sample_vehicles = df_clean['vehicle_id'].unique()[:50]
    for vid in sample_vehicles:
        veh_data = df_clean[df_clean['vehicle_id'] == vid]
        plt.plot(veh_data['x'], veh_data['y'], alpha=0.5)
    plt.title("Sample Vehicle Trajectories (First 50 Vehicles)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(results_figures / "raw_trajectories.png")
    plt.close()
    
    if 'speed' in df_clean.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df_clean['speed'].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title("Speed Distribution")
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Frequency")
        plt.savefig(results_figures / "speed_distribution.png")
        plt.close()
        
    print(f"Plots saved to {results_figures}/")
    print("\n--- PHASE 2 COMPLETE ---")

if __name__ == "__main__":
    preprocess_mobility_data()
