# wally — tiny, fast, append-only journal for Go

`wally` is a minimal append-only write-ahead log / journal with **zero-copy headers**, **optional in-memory indexing**, **crash-safe tail recovery**, and **batched writes**. It’s designed to be **small, simple, fast, and reliable** with as few allocations as possible.

* Append semantics with 1-based monotonic indexes
* On-disk record layout: `[len|crc32|reserved|payload]`
* Optional **full index** (O(1) random reads) or **sparse checkpoints** (tiny RAM)
* **Scan mode** (no index) for minimal memory usage
* **Batched appends** with single flush/fsync
* **Crash/torn-tail recovery** on open
* Concurrency-safe (single mutex); append order preserved

---

## Install

```bash
go get github.com/PlakarKorp/go-wally
```

```go
import "github.com/PlakarKorp/go-wally"
```

---

## Quick start

```go
l, err := wally.Open("events.wally", &wally.Options{
    NoSync:             true,    // don't fsync every append (call Sync/Close yourself)
    BufferSize:         256<<10, // default if <=0
    RetainIndex:        true,    // keep an index in memory
    CheckpointInterval: 4096,    // 1=full index; >1=sparse; 0 defaults to 4096
})
if err != nil { panic(err) }
defer l.Close()

// Append a record
idx, _ := l.Append([]byte("hello"))  // idx == 1

// Append a batch (one flush, one fsync if durable)
first, last, _ := l.AppendBatch([]byte("a"), []byte("b"), []byte("c"))

// Read it back
p, _ := l.Read(idx) // []byte("hello")

// Zero-alloc read (reuses dst if large enough)
dst := make([]byte, 0, 4096)
dst, _ = l.ReadInto(2, dst[:0])

// Truncate back to index N (keep 1..N)
_ = l.TruncateBack(3)

// Durability
_ = l.Sync()  // flush + fsync (unless NoSync)
_ = l.Close() // Close calls Sync under the hood
```

---

## Options & modes

| Option               | Meaning                                                                                          | Typical values                                           |
| -------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| `NoSync`             | If `true`, `Append`/`AppendBatch` won’t call `fsync`. Call `Sync`/`Close` for durability.        | `true` for throughput; `false` for per-append durability |
| `BufferSize`         | Size of the internal `bufio.Writer`                                                              | `256 KiB` default                                        |
| `RetainIndex`        | Keep an in-memory index for random reads                                                         | `true` or `false`                                        |
| `CheckpointInterval` | When indexing is enabled: `1`=full index (O(1)); `>1`=sparse checkpoints; `0` defaults to `4096` | `4096` is a good sparse default                          |

### Memory vs. random-read latency

* **Full index** (`RetainIndex=true`, `CheckpointInterval=1`)
  O(1) reads; memory ≈ `8 bytes × N`. For 10M records → ~76 MiB.

* **Sparse checkpoints** (`RetainIndex=true`, `CheckpointInterval=K>1`)
  Keep every K-th **header** offset. Memory ≈ `8 × ceil(N/K)` bytes.
  With K=4096 and N=10M → ~19 KB (!). Random read scans ≤K−1 headers (≤~64 KiB).

* **Scan mode** (`RetainIndex=false`)
  Minimal RAM, but random `Read(i)` scans from BOF (O(i)). Great for streaming/iteration.

---

## Record format (on disk)

Little-endian:

```
[0..7]   uint64 length
[8..11]  uint32 CRC32 (IEEE) of payload
[12..15] uint32 reserved (0)
[16..]   payload bytes
```

On open, `wally` scans the file, verifies CRCs, and **truncates** any torn tail at the first invalid record.

---

## API overview

```go
Open(path string, opts *Options) (*Log, error)
(*Log) Append(data []byte) (index uint64, err error)
(*Log) AppendBatch(records ...[]byte) (first, last uint64, err error)

type Batch struct {
    // Add/Reset/Len/Records to reuse allocations across batches
}

(*Log) WriteBatch(b *Batch) (first, last uint64, err error)

(*Log) Read(index uint64) ([]byte, error)           // allocates
(*Log) ReadInto(index uint64, dst []byte) ([]byte, error) // zero-alloc if cap(dst) >= len

(*Log) LastIndex() uint64
(*Log) TruncateBack(index uint64) error  // keep [1..index]; 0 clears file

(*Log) Sync() error
(*Log) Close() error  // calls Sync unless NoSync=true
```

Edge cases:

* Zero-length payloads are supported.
* `Read` on out-of-range index returns `io.EOF`.

---

## Durability & performance

* **Per-append durability** (`NoSync=false`): each `Append` fsyncs → latency dominated by filesystem (~ms).
  Use **`AppendBatch`** / `WriteBatch` to fsync **once** per group and achieve high durable throughput.

* **High throughput** (`NoSync=true`): rely on the OS page cache for speed; call `Sync` periodically (or `Close`) to persist.

* **Batched writes**: building headers + payloads and flushing once removes syscall overhead and reduces mutex contention.

---

## Concurrency

* The log is **concurrency-safe**; a single mutex preserves append order and guards internal state.
* Reads and writes are serialized by design for simplicity.
  If you need higher parallel read throughput, you can evolve to an `RWMutex` and per-call header buffers (reads under `RLock`).

---

## Benchmarks (example)

On an Apple M4 Pro (macOS, APFS), sample results:

```
Append (NoSync, 4KiB): ~1.06 GB/s, 0 allocs/op
AppendBatch (NoSync, 4KiB×64): ~3.88–4.33 GB/s, 0 allocs/op
Append (Sync each): ~4.3–4.6 ms/op (fsync bound)
ReadInto (4KiB): ~1.2 µs/op, ~3.4 GB/s, 0 allocs/op
```

Your numbers will depend on disk, filesystem, and options. Use:

```bash
go test -bench . -benchmem
```

to run the included microbenchmarks.

---

## Tuning tips

* **Durable throughput goal**: batch until **0.5–2 MiB per fsync**; you’ll keep ~4–6 ms latency but boost MB/s significantly.
* **Random reads on huge logs**: `RetainIndex=true`, `CheckpointInterval=4096` gives tiny RAM usage with fast seeks.
* **Minimal RAM**: `RetainIndex=false`. Reads scan; great for iterating the log.

---

## Testing

The repo includes comprehensive tests:

* Append/read round-trip across all modes
* Zero-length payloads
* Reopen + recovery (torn tail)
* Truncate semantics (including to zero) and post-truncate appends
* `ReadInto` buffer reuse (zero allocations)
* Durable mode (NoSync=false)
* Batch APIs (`AppendBatch`, `WriteBatch`)
* Sparse checkpoint boundaries

Run:

```bash
go test ./... -v
```

---

## FAQ

**Q: Why CRC32 and not xxhash?**
CRC32 IEEE is ubiquitous and hardware-accelerated on many platforms; it’s enough to catch torn writes and simple corruption. Swapable if needed.

**Q: Can I iterate without knowing indexes?**
Yes—just call `Read(1..LastIndex())` in a loop; in scan mode this is effectively a streaming scan.

**Q: Can I mmap or segment?**
Out of scope by default to keep the package tiny. A segmented variant with per-segment `uint32` indexes is a natural extension if you need O(1) reads with ~MB RAM.

**Q: Does `Close` call `Sync`?**
Yes. `Close` flushes and fsyncs unless `NoSync` is set.

---

## License

ISC

---

## Minimal example program

```go
package main

import (
	"fmt"
	"log"

	"github.com/PlakarKorp/go-wally"
)

func main() {
	l, err := wally.Open("demo.wally", &wally.Options{
		NoSync:             true,
		RetainIndex:        true,
		CheckpointInterval: 4096,
	})
	if err != nil { log.Fatal(err) }
	defer l.Close()

	first, last, err := l.AppendBatch([]byte("one"), []byte("two"), []byte("three"))
	if err != nil { log.Fatal(err) }
	fmt.Println("wrote indexes:", first, "to", last)

	p, _ := l.Read(2)
	fmt.Printf("index 2: %q\n", p)

	_ = l.Sync()
}
```
