def calculate_temporal_overlap(vids_current, vids_next):
    """
    Calculate common vehicle count between consecutive windows.
    """
    set_current = set(vids_current)
    set_next = set(vids_next)
    
    common = set_current.intersection(set_next)
    
    return {
        'common_vehicle_count': len(common),
        'total_current_vehicles': len(set_current),
        'total_next_vehicles': len(set_next)
    }
