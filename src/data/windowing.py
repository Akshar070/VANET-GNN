import os
import pandas as pd
from pathlib import Path

def create_time_windows(df, window_size, step_size, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_time = df['time'].min()
    max_time = df['time'].max()
    
    start_time = min_time
    
    metadata = []
    snapshots = []
    window_id = 0
    empty_windows = 0
    
    while start_time <= max_time:
        end_time = start_time + window_size
        
        # [start_time, end_time)
        window_df = df[(df['time'] >= start_time) & (df['time'] < end_time)]
        
        if not window_df.empty:
            # Sort by time to ensure tail(1) is the latest observation in the window
            window_df = window_df.sort_values(by=['time', 'vehicle_id'])
            
            # Last observation per vehicle
            snapshot_df = window_df.groupby('vehicle_id').tail(1).copy()
            
            actual_min = window_df['time'].min()
            actual_max = window_df['time'].max()
            vehicle_count = len(snapshot_df)
            record_count = len(window_df)
            
            snapshot_df['window_id'] = window_id
            snapshots.append(snapshot_df)
            
            metadata.append({
                'window_id': window_id,
                'start_time': start_time,
                'end_time': end_time,
                'actual_min_time': actual_min,
                'actual_max_time': actual_max,
                'vehicle_count': vehicle_count,
                'record_count': record_count
            })
        else:
            empty_windows += 1
            metadata.append({
                'window_id': window_id,
                'start_time': start_time,
                'end_time': end_time,
                'actual_min_time': None,
                'actual_max_time': None,
                'vehicle_count': 0,
                'record_count': 0
            })
            
        start_time += step_size
        window_id += 1
        
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / "window_metadata.csv", index=False)
    
    print(f"Generated {window_id} windows, {empty_windows} are empty.")
    
    if snapshots:
        snapshots_df = pd.concat(snapshots, ignore_index=True)
        snapshots_df = snapshots_df[['window_id', 'vehicle_id', 'time', 'x', 'y', 'speed']]
        snapshots_df.to_csv(output_dir / "window_snapshots.csv", index=False)
        print(f"Saved window snapshots to {output_dir / 'window_snapshots.csv'}")
        return metadata_df, snapshots_df
    
    return metadata_df, pd.DataFrame()
