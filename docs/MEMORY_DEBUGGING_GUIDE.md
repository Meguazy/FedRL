# Memory Debugging Guide

Your RAM is still growing despite all the fixes. Here's how to diagnose and fix it.

## Step 1: Monitor Memory in Real-Time

### Quick Monitor
```bash
./monitor_memory.sh
```

This will show you:
- RSS (actual RAM usage)
- Delta (how much it's growing per interval)
- Threads (check for thread leaks)

### Watch for Patterns:
- **Steady growth**: Memory leak (something accumulating)
- **Spiky growth**: Large allocations during rounds (might be OK)
- **Growth then plateau**: Initial loading (normal)
- **Growth without plateau**: Definite leak

## Step 2: Identify What's Growing

### Check Process Memory
```bash
ps aux | grep "server/main.py" | grep -v grep
```

Note the RSS (6th column) over time.

### Check if it's PyTorch/CUDA
```python
# Add this to your server code temporarily
import torch
if torch.cuda.is_available():
    print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
```

## Step 3: Common Memory Leak Sources

### 1. **Circular References with Closures**

**Location**: [server/main.py:405-425](server/main.py#L405-L425)

The `handle_model_update` closure captures `updates` dict and may create circular references.

**Fix**: Clear the handler after collection
```python
# After _collect_model_updates returns
self.server.set_message_handler(MessageType.MODEL_UPDATE, None)
```

### 2. **WebSocket Message Buffers**

WebSockets might be buffering messages. Check:
```bash
# Count open file descriptors
lsof -p <server_pid> | wc -l

# Should be reasonable (<100 per node)
```

### 3. **Loguru/Logging Buffers**

If you have file logging enabled, logs might be buffering in memory.

**Check**: Look at log file sizes
```bash
du -sh logs/*.log
```

**Fix**: Rotate logs or use external rotation
```python
# In your logging setup
from loguru import logger
logger.add("server.log", rotation="100 MB", retention="3 days")
```

### 4. **Metrics Storage Cache**

**Location**: [server/storage/file_metrics_store.py:73](server/storage/file_metrics_store.py#L73)

The `_index_cache` dict grows with each run.

**Fix**: Add cache size limit
```python
# Limit cache to last 10 runs
if len(self._index_cache) > 10:
    # Remove oldest entries
    oldest_keys = list(self._index_cache.keys())[:-10]
    for key in oldest_keys:
        del self._index_cache[key]
```

### 5. **Training Samples Not Released**

Even though we delete samples, Python might not release memory back to OS immediately.

**Force garbage collection**:
```python
import gc
gc.collect()  # After deleting large objects
```

### 6. **Message Handler Accumulation**

If message handlers aren't being replaced properly, old handlers might accumulate.

**Check**: [server/communication/server_socket.py:129](server/communication/server_socket.py#L129)
```python
self.message_handlers: Dict[MessageType, Callable] = {}
```

**Fix**: Clear handlers after each round
```python
# After round completes
self.server.message_handlers.clear()
```

## Step 4: Add Memory Profiling

### Option A: Use the debug script
```bash
# Monitor for 5 minutes
python debug_memory.py <server_pid> monitor 5 300
```

### Option B: Add to your code
```python
import tracemalloc

# At start of _execute_round
tracemalloc.start()

# At end of _execute_round
current, peak = tracemalloc.get_traced_memory()
log.info(f"Memory: current={current/1024**2:.1f}MB peak={peak/1024**2:.1f}MB")
tracemalloc.stop()
```

### Option C: Use memory_profiler
```bash
pip install memory-profiler

# Decorate your function
from memory_profiler import profile

@profile
async def _execute_round(self, round_num: int):
    ...
```

## Step 5: Targeted Fixes

Based on what you find, here are specific fixes:

### If it's WebSocket buffers:
```python
# Add to server_socket.py after sending messages
await websocket.drain()  # Flush buffers
```

### If it's message handlers:
```python
# Clear old handlers before setting new ones
self.message_handlers[MessageType.MODEL_UPDATE] = None
gc.collect()
```

### If it's metrics cache:
```python
# Add to file_metrics_store.py
def _cleanup_old_caches(self):
    if len(self._index_cache) > 10:
        # Keep only recent runs
        sorted_keys = sorted(self._index_cache.keys())
        for key in sorted_keys[:-10]:
            del self._index_cache[key]
```

### If it's Python not releasing memory:
```python
# After cleanup in _execute_round
import ctypes
ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux only
```

## Step 6: Confirm the Fix

Run your monitor and verify:
1. Memory growth stops after a few rounds
2. Memory stabilizes at a plateau
3. Delta stays near 0 after initial rounds

## Quick Diagnostic Commands

```bash
# 1. Watch memory every 5 seconds
watch -n 5 'ps aux | grep server/main.py | grep -v grep | awk "{print \$6/1024\" MB\"}"'

# 2. Count Python objects
python -c "import gc; gc.collect(); print(f'Objects: {len(gc.get_objects()):,}')"

# 3. Check for zombie processes
ps aux | grep python | grep -E "(agg_|pos_|tac_)" | wc -l

# 4. Monitor file descriptors
lsof -p <server_pid> | wc -l
```

## Expected Memory Usage

For your system with 8 nodes:
- **Server**: 500MB - 1.5GB (depends on model size)
- **Per Node**: 300MB - 800MB (depends on training data)
- **Total**: ~3-6GB for entire system

If you're seeing >10GB, there's definitely a leak.

## Next Steps

1. Run `./monitor_memory.sh` while training
2. Note when memory jumps (which step in the round)
3. Check the specific code in that step
4. Add targeted cleanup
5. Re-test

Let me know what the monitor shows and I can provide more specific fixes!
