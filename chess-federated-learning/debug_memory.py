#!/usr/bin/env python3
"""
Memory profiling utility for debugging memory leaks in the FL system.

Usage:
    python debug_memory.py <pid>
"""

import sys
import psutil
import gc
import tracemalloc
import time
from collections import Counter

def get_process_memory(pid):
    """Get memory info for a process."""
    try:
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()

        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': mem_percent,
            'num_threads': process.num_threads()
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"Error accessing process {pid}: {e}")
        return None

def monitor_memory(pid, interval=5, duration=300):
    """Monitor memory usage over time."""
    print(f"Monitoring process {pid} for {duration} seconds (sampling every {interval}s)")
    print(f"{'Time':<8} | {'RSS (MB)':<10} | {'VMS (MB)':<10} | {'% MEM':<8} | {'Threads':<8} | {'Delta RSS':<10}")
    print("-" * 80)

    start_time = time.time()
    previous_rss = None

    while time.time() - start_time < duration:
        mem = get_process_memory(pid)
        if mem is None:
            print(f"Process {pid} no longer exists")
            break

        elapsed = int(time.time() - start_time)
        delta = ""
        if previous_rss is not None:
            delta_mb = mem['rss_mb'] - previous_rss
            delta = f"+{delta_mb:.1f} MB" if delta_mb > 0 else f"{delta_mb:.1f} MB"
        previous_rss = mem['rss_mb']

        print(f"{elapsed:<8} | {mem['rss_mb']:<10.1f} | {mem['vms_mb']:<10.1f} | "
              f"{mem['percent']:<8.1f} | {mem['num_threads']:<8} | {delta:<10}")

        time.sleep(interval)

def get_object_counts():
    """Get counts of objects in memory by type."""
    gc.collect()

    # Count all objects
    all_objects = gc.get_objects()
    type_counts = Counter(type(obj).__name__ for obj in all_objects)

    print(f"\nTop 20 object types in memory:")
    print(f"{'Type':<40} | {'Count':<10}")
    print("-" * 55)

    for obj_type, count in type_counts.most_common(20):
        print(f"{obj_type:<40} | {count:<10,}")

    return type_counts

def find_large_objects(threshold_mb=1):
    """Find objects larger than threshold."""
    import sys
    gc.collect()

    print(f"\nLarge objects (>{threshold_mb} MB):")
    print(f"{'Type':<40} | {'Size (MB)':<10} | {'ID':<20}")
    print("-" * 75)

    threshold_bytes = threshold_mb * 1024 * 1024
    large_objects = []

    for obj in gc.get_objects():
        try:
            size = sys.getsizeof(obj)
            if size > threshold_bytes:
                large_objects.append((type(obj).__name__, size / 1024 / 1024, id(obj)))
        except:
            pass

    # Sort by size
    large_objects.sort(key=lambda x: x[1], reverse=True)

    for obj_type, size_mb, obj_id in large_objects[:20]:
        print(f"{obj_type:<40} | {size_mb:<10.2f} | {obj_id:<20}")

def check_pytorch_memory():
    """Check PyTorch memory if available."""
    try:
        import torch

        if torch.cuda.is_available():
            print("\nPyTorch CUDA Memory:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                print(f"  GPU {i}: Allocated={allocated:.1f}MB, Reserved={reserved:.1f}MB")
        else:
            print("\nPyTorch: No CUDA devices available")

    except ImportError:
        print("\nPyTorch not available")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_memory.py <pid> [command]")
        print("\nCommands:")
        print("  monitor [interval] [duration]  - Monitor memory over time (default: 5s, 300s)")
        print("  snapshot                       - Take a memory snapshot")
        print("  objects                        - Count objects by type")
        print("  large [threshold_mb]           - Find large objects")
        sys.exit(1)

    pid = int(sys.argv[1])
    command = sys.argv[2] if len(sys.argv) > 2 else "monitor"

    if command == "monitor":
        interval = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        duration = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        monitor_memory(pid, interval, duration)

    elif command == "snapshot":
        mem = get_process_memory(pid)
        if mem:
            print(f"Memory Snapshot for PID {pid}:")
            print(f"  RSS: {mem['rss_mb']:.1f} MB")
            print(f"  VMS: {mem['vms_mb']:.1f} MB")
            print(f"  Memory %: {mem['percent']:.1f}%")
            print(f"  Threads: {mem['num_threads']}")

    elif command == "objects":
        get_object_counts()

    elif command == "large":
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        find_large_objects(threshold)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
