import os
import re
import csv
import pandas as pd
from pathlib import Path
from collections import Counter

def parse_mobility_tcl(input_file="data/raw/mobility.tcl", output_file="data/interim/mobility_parsed.csv"):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    blank_lines = 0
    comment_lines = 0
    parsed_lines = 0
    skipped_lines = 0
    malformed_lines = 0
    
    pattern = re.compile(
        r'^\s*\$ns_\s+at\s+([0-9.]+)\s+"\$node_\((\d+)\)\s+setdest\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)"\s*$'
    )
    
    records = []
    
    print(f"Parsing {input_file} ...")
    
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, start=1):
            total_lines += 1
            line_str = line.strip()
            
            if not line_str:
                blank_lines += 1
                continue
                
            if line_str.startswith('#'):
                comment_lines += 1
                continue
                
            match = pattern.match(line_str)
            if match:
                time_val = float(match.group(1))
                vid = int(match.group(2))
                x_val = float(match.group(3))
                y_val = float(match.group(4))
                speed_val = float(match.group(5))
                
                records.append((time_val, vid, x_val, y_val, speed_val, i))
                parsed_lines += 1
            elif "$ns_ at" in line_str and "setdest" in line_str:
                malformed_lines += 1
            else:
                skipped_lines += 1
                
    if not records:
        raise ValueError("No valid mobility records found in the file.")
        
    print("Sorting records...")
    # Sort deterministically by time then vehicle_id
    records.sort(key=lambda r: (r[0], r[1]))
    
    seen = set()
    duplicate_count = 0
    for r in records:
        key = (r[0], r[1])
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
            
    vehicle_counts = Counter(r[1] for r in records)
    time_counts = Counter(r[0] for r in records)
    
    unique_vehicles = len(vehicle_counts)
    unique_timestamps = len(time_counts)
    min_time = min(r[0] for r in records)
    max_time = max(r[0] for r in records)
    min_x = min(r[2] for r in records)
    max_x = max(r[2] for r in records)
    min_y = min(r[3] for r in records)
    max_y = max(r[3] for r in records)
    min_s = min(r[4] for r in records)
    max_s = max(r[4] for r in records)
    
    avg_records_per_vehicle = sum(vehicle_counts.values()) / len(vehicle_counts)
    avg_records_per_time = sum(time_counts.values()) / len(time_counts)

    print("Writing CSV...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'vehicle_id', 'x', 'y', 'speed', 'source_line'])
        for r in records:
            writer.writerow(r)
            
    print("\n" + "="*42)
    print("VANET MOBILITY PARSER")
    print("="*42)
    print(f"\nInput:\n    {input_file}")
    print(f"\nOutput:\n    {output_file}")
    print(f"\nTotal lines:\n    {total_lines}")
    print(f"\nParsed mobility records:\n    {parsed_lines}")
    print(f"\nUnique vehicles:\n    {unique_vehicles}")
    print(f"\nTime range:\n    {min_time} -> {max_time}")
    print(f"\nUnique timestamps:\n    {unique_timestamps}")
    print(f"\nCoordinate range:\n    X: {min_x} -> {max_x}\n    Y: {min_y} -> {max_y}")
    print(f"\nSpeed range:\n    {min_s} -> {max_s}")
    print(f"\nMalformed/skipped:\n    {malformed_lines + skipped_lines} (malformed: {malformed_lines}, skipped: {skipped_lines})")
    print(f"\nDuplicate records:\n    {duplicate_count}")
    print(f"\nRecords per vehicle:\n    Avg: {avg_records_per_vehicle:.2f}, Min: {min(vehicle_counts.values())}, Max: {max(vehicle_counts.values())}")
    print(f"\nRecords per timestamp:\n    Avg: {avg_records_per_time:.2f}, Min: {min(time_counts.values())}, Max: {max(time_counts.values())}")
    print("="*42)
    
    validate_parsed_data(output_path)

def validate_parsed_data(csv_path):
    print("\n--- POST-PARSING INSPECTION ---")
    df = pd.read_csv(csv_path)
    
    print("\nDataFrame Head:")
    print(df.head())
    
    print("\nDataFrame Tail:")
    print(df.tail())
    
    print(f"\nShape: {df.shape}")
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isna().sum())
    
    print(f"\nUnique Vehicle IDs: {df['vehicle_id'].nunique()}")
    print(f"Min Time: {df['time'].min()}")
    print(f"Max Time: {df['time'].max()}")
    
    times = sorted(df['time'].unique())
    if times:
        first_time = times[0]
        mid_time = times[len(times)//2]
        last_time = times[-1]
        
        print(f"\nRecords at First Timestamp ({first_time}):")
        print(df[df['time'] == first_time].head())
        
        print(f"\nRecords at Middle Timestamp ({mid_time}):")
        print(df[df['time'] == mid_time].head())
        
        print(f"\nRecords at Final Timestamp ({last_time}):")
        print(df[df['time'] == last_time].head())

if __name__ == "__main__":
    parse_mobility_tcl()
