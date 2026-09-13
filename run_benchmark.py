import time
import gc
import statistics
from your_filter_module import BitPerfectFilter  # Adjust import to match your repo

def run_advanced_benchmark():
    filter_obj = BitPerfectFilter()

    # Construct realistic text profiles mimicking log aggregation pipelines
    test_profiles = {
        "Short String": "short text",
        "Medium Log": "INFO 2026-09-12 23:29:01 [Worker-1] Database transaction completed successfully. " * 50,
        "Heavy Code Blob": "function entry() { console.log('system_trace_dump'); return Array(100).fill(0); }\n" * 500,
        "Deep JSON Payload": '{"status": "active", "metrics": {"cpu": 0.45, "mem": 0.88, "io": [12, 45, 78, 12, 0]}}, ' * 300
    }

    # Configuration Parameters
    WARMUP_RUNS = 5
    TEST_RUNS = 50

    print(f"{'Profile Name':<18} | {'Chars':>8} | {'Avg (ms)':>9} | {'StDev (ms)':>10} | {'Throughput':>12} | {'Suppressed'}")
    print("-" * 82)

    for name, text in test_profiles.items():
        # 1. Warm-up Phase: Forces CPU caching and runtime initialization
        for _ in range(WARMUP_RUNS):
            _ = filter_obj.filter_message(text)

        # 2. Environmental Control: Force GC to avoid middle-of-run collection spikes
        gc.collect()
        gc.disable()

        # 3. Execution Phase
        timings = []
        result = None
        
        for _ in range(TEST_RUNS):
            start = time.perf_counter()
            result = filter_obj.filter_message(text)
            end = time.perf_counter()
            timings.append((end - start) * 1000) # Convert to milliseconds

        gc.enable() # Restore garbage collection framework

        # 4. Statistical Metrics Aggregation
        avg_time = statistics.mean(timings)
        stdev_time = statistics.stdev(timings) if len(timings) > 1 else 0.0
        
        # Calculate Throughput: Megabytes processed per second
        total_chars = len(text)
        mb_processed = (total_chars * 1) / (1024 * 1024) # Assuming 1 byte per char ASCII
        sec_elapsed = avg_time / 1000
        throughput = f"{mb_processed / sec_elapsed:.2f} MB/s" if sec_elapsed > 0 else "N/A"

        print(f"{name:<18} | {total_chars:>8} | {avg_time:>9.3f} | {stdev_time:>10.3f} | {throughput:>12} | {str(result['is_suppressed']):<10}")

if __name__ == "__main__":
    run_advanced_benchmark()
