//go:build ignore

package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/PlakarKorp/go-wally"
)

func main() {
	var (
		path        = flag.String("path", "demo.wal", "output wally path")
		records     = flag.Int("records", 10_000_000, "number of records to write")
		recSize     = flag.Int("size", 1024, "record payload size in bytes")
		batchSize   = flag.Int("batch", 4096, "append batch size (records per batch)")
		retain      = flag.Bool("retain", false, "retain in-memory index")
		checkpoint  = flag.Int("k", 4096, "checkpoint interval when -retain is true (1=full index, >1=sparse)")
		noSync      = flag.Bool("nosync", true, "disable fsync on each append/batch (call Sync at end)")
		progress    = flag.Int("progress_every", 1_000_000, "print stats every N records")
		pattern     = flag.String("pattern", "zero", "payload pattern: zero | seqbyte (seqbyte requires -writers=1)")
		verifyAll   = flag.Bool("verify_all", false, "verify all records after writing (can be slow)")
		verifyN     = flag.Int("verify_samples", 1000, "number of evenly-spaced samples to verify if -verify_all=false")
		compression = flag.String("compression", "snappy", "wally compression: none | snappy")

		// NEW:
		writers = flag.Int("writers", 1, "number of parallel writer goroutines")
		queue   = flag.Int("queue", 0, "jobs queue depth (default: 2 * writers)")
	)
	flag.Parse()

	if *writers > 1 && *pattern == "seqbyte" {
		log.Fatalf("pattern=seqbyte is not supported with writers>1 (verification cannot be deterministic). Use -pattern=zero for parallel.")
	}
	if *batchSize <= 0 {
		log.Fatalf("batch must be > 0")
	}
	if *records <= 0 {
		log.Fatalf("records must be > 0")
	}
	if *queue <= 0 {
		*queue = 2 * *writers
	}

	opts := &wally.Options{
		NoSync:             *noSync,
		BufferSize:         256 << 10,
		RetainIndex:        *retain,
		CheckpointInterval: uint(*checkpoint),
		Compression:        *compression,
	}

	l, err := wally.Open(*path, opts)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		_ = l.Sync()
		_ = l.Close()
	}()

	headerSize := 16
	if *compression != "" && *compression != "none" {
		headerSize = 24
	}

	fmt.Printf(
		"Writing %d records of %d bytes to %s (batch=%d, retain=%v, k=%d, nosync=%v, compression=%s, pattern=%s, writers=%d, queue=%d)\n",
		*records, *recSize, *path, *batchSize, *retain, *checkpoint, *noSync, *compression, *pattern, *writers, *queue,
	)

	start := time.Now()
	lastMark := start

	if *writers == 1 {
		// -------- Single-threaded (original path) --------
		runSingleThread(l, *path, *records, *recSize, *batchSize, *noSync, *progress, *pattern, headerSize, start, &lastMark)
	} else {
		// -------- Parallel writers --------
		runParallel(l, *path, *records, *recSize, *batchSize, *writers, *queue, *noSync, *progress, headerSize, start, &lastMark)
	}

	// Final GC snapshot
	_ = l.Sync()
	runtime.GC()
	fmt.Println("After final GC:")
	printMem()

	// Report from actual on-disk size (accounts for compression)
	fileBytes := fileSize(*path)
	elapsed := time.Since(start).Seconds()
	fmt.Printf("DONE: wrote %d records; file size = %.2f GiB; elapsed = %.2fs; throughput = %.2f MiB/s\n",
		*records, float64(fileBytes)/(1<<30), elapsed, (float64(fileBytes)/(1<<20))/elapsed,
	)

	// --- Reopen & verify ---
	fmt.Println("\nReopening for verification...")
	if err := l.Close(); err != nil {
		log.Fatalf("close before reopen: %v", err)
	}
	l2, err := wally.Open(*path, &wally.Options{
		NoSync:             true,
		BufferSize:         256 << 10,
		RetainIndex:        true,
		CheckpointInterval: 0,            // full index
		Compression:        *compression, // fine either way
		//DeleteOnClose:    true,
	})
	if err != nil {
		log.Fatalf("reopen: %v", err)
	}
	defer l2.Close()

	lastIdx := l2.LastIndex()
	fmt.Printf("LastIndex: %d\n", lastIdx)

	startVerify := time.Now()
	var verified int
	// With parallel writers we only support "zero" verification (content is independent of index).
	if *verifyAll {
		verified = verifyRange(l2, 1, uint64(*records), *recSize, safePatternForVerify(*pattern, *writers))
	} else {
		verified = verifySamples(l2, *records, *recSize, safePatternForVerify(*pattern, *writers), *verifyN)
	}
	fmt.Printf("Verified %d record(s) in %.2fs — OK ✅\n", verified, time.Since(startVerify).Seconds())
}

// --------------------------------- helpers -----------------------------------

func runSingleThread(
	l *wally.Log,
	path string,
	records, recSize, batchSize int,
	noSync bool,
	progress int,
	pattern string,
	headerSize int,
	start time.Time,
	lastMark *time.Time,
) {
	written := 0

	// --- Zero-alloc batching setup ---
	zeroPayload := make([]byte, recSize)
	perSlot := make([][]byte, batchSize)
	for i := range perSlot {
		perSlot[i] = make([]byte, recSize)
	}
	b := make([][]byte, batchSize)

	for written < records {
		remain := records - written
		n := batchSize
		if remain < n {
			n = remain
		}

		switch pattern {
		case "zero":
			for i := 0; i < n; i++ {
				b[i] = zeroPayload
			}
		case "seqbyte":
			base := written + 1
			for i := 0; i < n; i++ {
				v := byte((base + i) % 256)
				buf := perSlot[i]
				for j := range buf {
					buf[j] = v
				}
				b[i] = buf
			}
		default:
			for i := 0; i < n; i++ {
				b[i] = zeroPayload
			}
		}

		if _, _, err := l.AppendBatch(b[:n]...); err != nil {
			log.Fatalf("AppendBatch: %v", err)
		}
		written += n

		if written%progress == 0 || written == records {
			if !noSync {
				_ = l.Sync()
			}
			printStatsFS(path, written, recSize, headerSize, start, *lastMark)
			*lastMark = time.Now()
		}
	}
}

type job struct {
	n int // number of records to write in this batch (<= batchSize)
	// Workers have their own preallocated slots/buffer slices, so no payloads here.
}

func runParallel(
	l *wally.Log,
	path string,
	records, recSize, batchSize, writers, qDepth int,
	noSync bool,
	progress int,
	headerSize int,
	start time.Time,
	lastMark *time.Time,
) {
	// Global counters
	var written atomic.Int64

	// Job channel
	jobs := make(chan job, qDepth)

	var wg sync.WaitGroup
	wg.Add(writers)

	// Start workers
	for wid := 0; wid < writers; wid++ {
		go func(wid int) {
			defer wg.Done()

			// --- Per-worker zero-alloc setup ---
			zeroPayload := make([]byte, recSize)
			perSlot := make([][]byte, batchSize)
			for i := range perSlot {
				perSlot[i] = make([]byte, recSize)
			}
			b := make([][]byte, batchSize)

			for j := range jobs {
				// For parallel mode we only support "zero" pattern (deterministic verification).
				n := j.n
				for i := 0; i < n; i++ {
					b[i] = zeroPayload
				}
				if _, _, err := l.AppendBatch(b[:n]...); err != nil {
					log.Fatalf("[worker %d] AppendBatch: %v", wid, err)
				}
				newWritten := written.Add(int64(n))

				// local opportunistic progress printing kept silent here
				_ = newWritten
			}
		}(wid)
	}

	// Progress/ticker goroutine (single printer)
	done := make(chan struct{})
	var prWG sync.WaitGroup
	prWG.Add(1)
	go func() {
		defer prWG.Done()
		t := time.NewTicker(1 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				w := int(written.Load())
				if w > 0 && (w%progress == 0 || w == records) {
					if !noSync {
						_ = l.Sync()
					}
					printStatsFS(path, w, recSize, headerSize, start, *lastMark)
					*lastMark = time.Now()
				}
			}
		}
	}()

	// Feed jobs (bounded by queue)
	remain := records
	for remain > 0 {
		n := batchSize
		if remain < n {
			n = remain
		}
		jobs <- job{n: n}
		remain -= n
	}
	close(jobs)

	// Wait for writers
	wg.Wait()
	// Stop progress goroutine and print final progress if needed
	close(done)
	prWG.Wait()

	// Final progress snapshot if we missed an exact boundary
	w := int(written.Load())
	if w != 0 && w != records {
		if !noSync {
			_ = l.Sync()
		}
		printStatsFS(path, w, recSize, headerSize, start, *lastMark)
		*lastMark = time.Now()
	}

	if w != records {
		log.Fatalf("written mismatch: got %d want %d", w, records)
	}
}

func safePatternForVerify(pattern string, writers int) string {
	if writers > 1 {
		return "zero"
	}
	return pattern
}

// --- verification helpers ----------------------------------------------------

func verifySamples(l *wally.Log, total, size int, pattern string, samples int) int {
	if samples <= 0 {
		samples = 1
	}
	step := math.Max(1, float64(total)/float64(samples))
	idx := 1.0
	dst := make([]byte, 0, size)
	count := 0
	for i := 0; i < samples; i++ {
		n := int(idx + 0.5)
		if n < 1 {
			n = 1
		} else if n > total {
			n = total
		}
		var err error
		dst, err = l.ReadInto(uint64(n), dst[:0])
		if err != nil {
			log.Fatalf("ReadInto(%d): %v", n, err)
		}
		checkPattern(uint64(n), dst, pattern)
		idx += step
		count++
	}
	return count
}

func verifyRange(l *wally.Log, from, to uint64, size int, pattern string) int {
	dst := make([]byte, 0, size)
	for i := from; i <= to; i++ {
		var err error
		dst, err = l.ReadInto(i, dst[:0])
		if err != nil {
			log.Fatalf("ReadInto(%d): %v", i, err)
		}
		checkPattern(i, dst, pattern)
	}
	return int(to - from + 1)
}

func checkPattern(idx uint64, buf []byte, pattern string) {
	switch pattern {
	case "seqbyte":
		want := byte(int(idx) % 256)
		for i := 0; i < len(buf); i++ {
			if buf[i] != want {
				log.Fatalf("content mismatch at %d: buf[%d]=%d want %d", idx, i, buf[i], want)
			}
		}
	case "zero":
		for i := 0; i < len(buf); i++ {
			if buf[i] != 0 {
				log.Fatalf("content mismatch at %d: buf[%d]=%d want 0", idx, i, buf[i])
			}
		}
	default:
		log.Fatalf("unknown pattern %q", pattern)
	}
}

// --- stats helpers -----------------------------------------------------------

func printStatsFS(path string, written, recSize, headerSize int, start, last time.Time) {
	fileBytes := fileSize(path)
	elapsed := time.Since(start).Seconds()
	fmt.Printf("\n== progress: %d records ==\n", written)
	printMem()
	fmt.Printf("File size now: %.2f GiB | Since start: %.2fs | Throughput: %.2f MiB/s\n",
		float64(fileBytes)/(1<<30),
		elapsed,
		(float64(fileBytes)/(1<<20))/elapsed,
	)
	// compression ratio (file bytes vs logical header+payload)
	logical := int64(written) * int64(headerSize+recSize)
	if logical > 0 && fileBytes > 0 {
		fmt.Printf("Compression ratio (file/logical): %.3fx\n", float64(fileBytes)/float64(logical))
	}
}

func printMem() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Mem: Alloc=%.2f MiB  TotalAlloc=%.2f MiB  Sys=%.2f MiB  NumGC=%d\n",
		float64(m.Alloc)/(1<<20),
		float64(m.TotalAlloc)/(1<<20),
		float64(m.Sys)/(1<<20),
		m.NumGC)
}

func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}
