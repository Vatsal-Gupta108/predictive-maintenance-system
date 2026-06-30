import os
import numpy as np
import pandas as pd

def generate_predictive_maintenance_data(seed=42):
    np.random.seed(seed)
    
    # Configuration
    num_machines = 5
    num_days = 365
    hours_per_day = 24
    total_hours = num_days * hours_per_day  # 8760 hours per machine
    
    # Initialize lists to hold dataset records
    telemetry_records = []
    maintenance_records = []
    
    print(f"Starting simulation for {num_machines} machines over {num_days} days...")
    
    for machine_id in range(1, num_machines + 1):
        # Base healthy parameters for this machine
        base_temp = np.random.uniform(55, 65)  # Normal operating temp in C
        base_pressure = np.random.uniform(180, 220)  # Normal pressure in kPa
        base_vibration = np.random.uniform(1.0, 1.5)  # Normal vibration in mm/s
        base_voltage = 220.0  # Volts
        
        # Wear accumulation tracking
        wear_accum = 0.0
        tool_wear = 0.0
        consecutive_overstrain_hours = 0
        hdf_active = False
        hdf_hours = 0
        
        # Time tracking
        t = 0
        while t < total_hours:
            # Current timestamp
            timestamp = pd.Timestamp("2026-01-01") + pd.to_timedelta(t, unit="h")
            
            # Daily cycles (ambient effects)
            hour_of_day = t % 24
            cycle_effect = np.sin(2 * np.pi * hour_of_day / 24)
            
            # Base wear increases slowly with time
            wear_accum += np.random.exponential(0.0005)
            tool_wear += np.random.uniform(0.01, 0.03)  # Tool wear creeps up
            
            # Sensors under normal conditions
            voltage = base_voltage + np.random.normal(0, 5.0)
            speed = 1500.0 + cycle_effect * 50.0 + np.random.normal(0, 25.0)
            
            # Wear increases temperature and vibration
            temperature = base_temp + (wear_accum * 15.0) + cycle_effect * 2.0 + np.random.normal(0, 1.0)
            vibration = base_vibration + (wear_accum * 1.2) + np.random.normal(0, 0.15)
            pressure = base_pressure - (wear_accum * 8.0) + np.random.normal(0, 5.0)
            
            # Check and simulate failures
            is_failure = False
            failure_type = "None"
            
            # 1. Simulate Heat Dissipation Failure (HDF) trigger
            # 1.5% chance to start HDF sequence if healthy
            if not hdf_active and np.random.random() < 0.001 and wear_accum > 0.1:
                hdf_active = True
                hdf_hours = 0
                
            if hdf_active:
                hdf_hours += 1
                # Temperature spikes rapidly
                temperature += (hdf_hours ** 1.5) * 1.5
                vibration += (hdf_hours * 0.08)
                if temperature >= 95.0:
                    is_failure = True
                    failure_type = "HDF"
                    hdf_active = False
            
            # 2. Simulate Power Failure (PWF)
            # Sudden voltage spike or drop
            if not is_failure and np.random.random() < 0.0005:
                voltage = 220.0 + np.random.choice([-45.0, 45.0]) + np.random.normal(0, 2)
                if voltage < 180.0 or voltage > 260.0:
                    is_failure = True
                    failure_type = "PWF"
            
            # 3. Simulate Overstrain Failure (OSF)
            # High speed + high vibration product
            stress_index = (speed / 1000.0) * vibration
            if stress_index > 2.8:
                consecutive_overstrain_hours += 1
            else:
                consecutive_overstrain_hours = max(0, consecutive_overstrain_hours - 1)
                
            if not is_failure and consecutive_overstrain_hours >= 3:
                is_failure = True
                failure_type = "OSF"
                consecutive_overstrain_hours = 0
                
            # 4. Simulate Tool Wear Failure (TWF)
            if not is_failure and tool_wear > 100.0:
                is_failure = True
                failure_type = "TWF"
                
            # Handle Failure event and Maintenance Reset
            if is_failure:
                # Add to maintenance records
                maintenance_records.append({
                    "timestamp": timestamp,
                    "machine_id": f"M_{machine_id:03d}",
                    "event_type": "Failure",
                    "failure_type": failure_type,
                    "action": "Repair & Replacement"
                })
                
                # Append telemetry at point of failure
                telemetry_records.append({
                    "timestamp": timestamp,
                    "machine_id": f"M_{machine_id:03d}",
                    "voltage": voltage,
                    "temperature": temperature,
                    "vibration": vibration,
                    "pressure": pressure,
                    "rotational_speed": speed,
                    "tool_wear": tool_wear
                })
                
                # Machine goes offline for repair (e.g. 4 hours)
                for offline_hour in range(1, 5):
                    t += 1
                    offline_timestamp = timestamp + pd.to_timedelta(offline_hour, unit="h")
                    # Offline values are baselines/zeroed out
                    telemetry_records.append({
                        "timestamp": offline_timestamp,
                        "machine_id": f"M_{machine_id:03d}",
                        "voltage": 0.0,
                        "temperature": 20.0,  # Ambient cooled
                        "vibration": 0.0,
                        "pressure": 0.0,
                        "rotational_speed": 0.0,
                        "tool_wear": 0.0
                    })
                
                # Reset degradation counters post-repair
                wear_accum = 0.0
                tool_wear = 0.0
                consecutive_overstrain_hours = 0
                hdf_active = False
                
            else:
                # Normal operational telemetry recording
                telemetry_records.append({
                    "timestamp": timestamp,
                    "machine_id": f"M_{machine_id:03d}",
                    "voltage": voltage,
                    "temperature": temperature,
                    "vibration": vibration,
                    "pressure": pressure,
                    "rotational_speed": speed,
                    "tool_wear": tool_wear
                })
                
                # 5. Occasional Routine Maintenance (Preventative)
                # Crew replaces parts routine-wise every ~700 hours (roughly 1 month)
                if t > 0 and t % 720 == 0 and np.random.random() < 0.8:
                    maintenance_records.append({
                        "timestamp": timestamp,
                        "machine_id": f"M_{machine_id:03d}",
                        "event_type": "Routine Maintenance",
                        "failure_type": "None",
                        "action": "Inspection & Calibration"
                    })
                    # Reduce wear by 80% (restores machine condition)
                    wear_accum *= 0.2
                    tool_wear = 0.0
            
            t += 1
            
    # Convert to DataFrames
    df_telemetry = pd.DataFrame(telemetry_records)
    df_maintenance = pd.DataFrame(maintenance_records)
    
    return df_telemetry, df_maintenance

if __name__ == "__main__":
    # Create target directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Generate datasets
    df_tel, df_maint = generate_predictive_maintenance_data(seed=42)
    
    # Save to CSV
    tel_path = "data/raw/sensor_telemetry.csv"
    maint_path = "data/raw/maintenance_log.csv"
    
    df_tel.to_csv(tel_path, index=False)
    df_maint.to_csv(maint_path, index=False)
    
    print("\n--- Simulation Summary ---")
    print(f"Telemetry records saved to: {tel_path} ({df_tel.shape[0]} rows)")
    print(f"Maintenance logs saved to: {maint_path} ({df_maint.shape[0]} rows)")
    print("\nBreakdown of maintenance events:")
    print(df_maint["event_type"].value_counts())
    print("\nBreakdown of failure modes:")
    print(df_maint[df_maint["event_type"] == "Failure"]["failure_type"].value_counts())
