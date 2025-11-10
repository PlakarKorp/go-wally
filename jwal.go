package jwal

import (
	"bufio"
	"encoding/binary"
	"hash/crc32"
	"io"
	"os"
	"sync"
	"sync/atomic"

	"github.com/golang/snappy"
)

// Record layout (little-endian, 24-byte header):
// [0..7]   uint64 storedLen         // number of bytes stored on disk (after compression if any)
// [8..11]  uint32 crc32(stored)     // checksum of the stored bytes; used for torn-tail recovery
// [12..15] uint32 uncompressedLen   // length of payload before compression; 0 if uncompressed
// [16..19] uint32 codec             // 0=none, 1=snappy
// [20..23] uint32 reserved
// [24..]   payload (compressed if codec!=none)
const headerSize = 24

const (
	codecNone   = 0
	codecSnappy = 1
)

type Options struct {
	NoSync             bool // if true, Append won't fsync; call Sync manually
	BufferSize         int  // bufio size; default 256 KiB if <=0
	RetainIndex        bool // if true, keep an index (full or sparse)
	CheckpointInterval uint // when RetainIndex=true:
	//   1 => keep every record offset (full index, ~8B/record)
	//   >1 => keep every K-th record header offset (sparse)
	//   0 => defaults to 4096 (sparse)

	// Compression: "none" (default) or "snappy"
	Compression   string
	DeleteOnClose bool
}

type Log struct {
	mu   sync.Mutex
	fp   *os.File
	w    *bufio.Writer
	path string

	idx []int64

	// config
	noSync             bool
	retainIndex        bool
	checkpointInterval uint
	deleteOnClose      bool
	codec              uint32 // codecNone or codecSnappy

	// counters
	count atomic.Uint64

	// scratch buffers (used under mu)
	compScratch []byte
}

// Open opens or creates the journal and scans to recover the tail.
func Open(path string, opts *Options) (*Log, error) {
	if opts == nil {
		opts = &Options{}
	}

	var codec uint32
	switch opts.Compression {
	case "", "none":
		codec = codecNone
	case "snappy":
		codec = codecSnappy
	default:
		return nil, io.ErrUnexpectedEOF
	}

	fp, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}

	bufsz := opts.BufferSize
	if bufsz <= 0 {
		bufsz = 256 << 10
	}

	l := &Log{
		fp:                 fp,
		path:               path,
		w:                  bufio.NewWriterSize(fp, bufsz),
		noSync:             opts.NoSync,
		retainIndex:        opts.RetainIndex,
		checkpointInterval: opts.CheckpointInterval,
		deleteOnClose:      opts.DeleteOnClose,
		codec:              codec,
	}
	if l.retainIndex {
		if l.checkpointInterval == 0 {
			l.checkpointInterval = 4096
		}
		if l.checkpointInterval < 1 {
			l.checkpointInterval = 1
		}
	}

	if err := l.scanAndRecover(); err != nil {
		_ = fp.Close()
		return nil, err
	}
	return l, nil
}

// scanAndRecover validates stored bytes by crc32(stored) and truncates torn tails.
// It also rebuilds the (full or sparse) in-memory index.
func (l *Log) scanAndRecover() error {
	var off int64 // header offset of current record
	var buf []byte
	var hdr [headerSize]byte

	fi, err := l.fp.Stat()
	if err != nil {
		return err
	}
	size := fi.Size()

	for off+headerSize <= size {
		// read header
		if _, err := l.fp.ReadAt(hdr[:], off); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		storedLen := binary.LittleEndian.Uint64(hdr[0:8])
		end := off + headerSize + int64(storedLen)
		if end > size {
			return l.fp.Truncate(off)
		}
		wantStoredCRC := binary.LittleEndian.Uint32(hdr[8:12])

		// verify checksum over stored bytes (fast, no decompression needed)
		if len(buf) < int(storedLen) {
			buf = make([]byte, int(storedLen))
		}
		if _, err := l.fp.ReadAt(buf[:int(storedLen)], off+headerSize); err != nil {
			if err == io.EOF {
				return l.fp.Truncate(off)
			}
			return err
		}
		got := crc32.ChecksumIEEE(buf[:int(storedLen)])
		if got != wantStoredCRC {
			return l.fp.Truncate(off)
		}

		// index
		if l.retainIndex {
			if l.checkpointInterval == 1 {
				// full index stores DATA-start offsets (payload start)
				l.idx = append(l.idx, off+headerSize)
			} else {
				recNum := l.count.Load() + 1
				if (recNum-1)%uint64(l.checkpointInterval) == 0 {
					// sparse stores HEADER offsets at checkpoints
					l.idx = append(l.idx, off)
				}
			}
		}
		l.count.Add(1)
		off = end
	}
	_, err = l.fp.Seek(0, io.SeekEnd)
	return err
}

// Append appends a new record (compressed if enabled) and returns its 1-based index.
func (l *Log) Append(data []byte) (uint64, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	var hdr [headerSize]byte

	hdrOff, err := l.fp.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, err
	}

	// Prepare payload (maybe compressed)
	payload, storedLen, ulen, codec := l.preparePayloadLocked(data)

	// Write header
	binary.LittleEndian.PutUint64(hdr[0:8], storedLen)
	binary.LittleEndian.PutUint32(hdr[8:12], crc32.ChecksumIEEE(payload))
	binary.LittleEndian.PutUint32(hdr[12:16], ulen)
	binary.LittleEndian.PutUint32(hdr[16:20], codec)
	// [20..23] reserved = 0

	if _, err := l.w.Write(hdr[:]); err != nil {
		return 0, err
	}
	if _, err := l.w.Write(payload); err != nil {
		return 0, err
	}
	if err := l.w.Flush(); err != nil {
		return 0, err
	}
	if !l.noSync {
		if err := l.fp.Sync(); err != nil {
			return 0, err
		}
	}

	// update index
	if l.retainIndex {
		if l.checkpointInterval == 1 {
			l.idx = append(l.idx, hdrOff+headerSize) // payload start
		} else {
			next := l.count.Load() + 1
			if (next-1)%uint64(l.checkpointInterval) == 0 {
				l.idx = append(l.idx, hdrOff) // header checkpoint
			}
		}
	}

	l.count.Add(1)
	return l.count.Load(), nil
}

func (l *Log) preparePayloadLocked(data []byte) (payload []byte, storedLen uint64, ulen uint32, codec uint32) {
	switch l.codec {
	case codecNone:
		return data, uint64(len(data)), 0, codecNone
	case codecSnappy:
		l.compScratch = snappy.Encode(l.compScratch[:0], data)
		return l.compScratch, uint64(len(l.compScratch)), uint32(len(data)), codecSnappy
	default:
		return data, uint64(len(data)), 0, codecNone
	}
}

func (l *Log) Read(index uint64) ([]byte, error) {
	if index == 0 || index > l.count.Load() {
		return nil, io.EOF
	}

	var hdr [headerSize]byte

	dataOff, err := l.locateDataOffsetLocked(index)
	if err != nil {
		return nil, err
	}

	// read header again (dataOff-headerSize)
	if _, err := l.fp.ReadAt(hdr[:], dataOff-headerSize); err != nil {
		return nil, err
	}
	storedLen := int(binary.LittleEndian.Uint64(hdr[0:8]))
	ulen := binary.LittleEndian.Uint32(hdr[12:16])
	codec := binary.LittleEndian.Uint32(hdr[16:20])

	// read stored bytes
	buf := make([]byte, storedLen)
	if _, err := l.fp.ReadAt(buf, dataOff); err != nil {
		return nil, err
	}

	// verify stored CRC (lightweight)
	if crc32.ChecksumIEEE(buf) != binary.LittleEndian.Uint32(hdr[8:12]) {
		return nil, io.ErrUnexpectedEOF
	}

	// decompress if needed
	switch codec {
	case codecNone:
		return buf, nil
	case codecSnappy:
		out, err := snappy.Decode(nil, buf)
		if err != nil {
			return nil, err
		}
		// optional: sanity-check uncompressed length if present
		if ulen != 0 && uint32(len(out)) != ulen {
			return nil, io.ErrUnexpectedEOF
		}
		return out, nil
	default:
		return nil, io.ErrUnexpectedEOF
	}
}

func (l *Log) ReadInto(index uint64, dst []byte) ([]byte, error) {
	if index == 0 || index > l.count.Load() {
		return nil, io.EOF
	}

	var hdr [headerSize]byte

	dataOff, err := l.locateDataOffsetLocked(index)
	if err != nil {
		return nil, err
	}
	if _, err := l.fp.ReadAt(hdr[:], dataOff-headerSize); err != nil {
		return nil, err
	}
	storedLen := int(binary.LittleEndian.Uint64(hdr[0:8]))
	ulen := binary.LittleEndian.Uint32(hdr[12:16])
	codec := binary.LittleEndian.Uint32(hdr[16:20])

	// read stored bytes
	tmp := make([]byte, storedLen)
	if _, err := l.fp.ReadAt(tmp, dataOff); err != nil {
		return nil, err
	}
	if crc32.ChecksumIEEE(tmp) != binary.LittleEndian.Uint32(hdr[8:12]) {
		return nil, io.ErrUnexpectedEOF
	}

	switch codec {
	case codecNone:
		// copy into dst if provided; ReadInto contract returns owned slice
		n := len(tmp)
		if cap(dst) < n {
			dst = make([]byte, n)
		} else {
			dst = dst[:n]
		}
		copy(dst, tmp)
		return dst, nil
	case codecSnappy:
		// snappy.Decode requires a new buffer; we can allocate exact size if ulen>0
		if ulen > 0 && int(ulen) <= cap(dst) {
			dst = dst[:int(ulen)]
			out, err := snappy.Decode(dst[:0], tmp) // will reuse provide dst backing
			if err != nil {
				return nil, err
			}
			return out, nil
		}
		out, err := snappy.Decode(nil, tmp)
		if err != nil {
			return nil, err
		}
		if ulen != 0 && uint32(len(out)) != ulen {
			return nil, io.ErrUnexpectedEOF
		}
		return out, nil
	default:
		return nil, io.ErrUnexpectedEOF
	}
}

func (l *Log) LastIndex() uint64 {
	return l.count.Load()
}

// TruncateBack keeps records up to and including index; if index==0 clears file.
func (l *Log) TruncateBack(index uint64) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if index > l.count.Load() {
		return io.EOF
	}

	var hdr [headerSize]byte

	var newSize int64
	if index == 0 {
		newSize = 0
	} else {
		dataOff, err := l.locateDataOffsetLocked(index)
		if err != nil {
			return err
		}
		// read header to get storedLen
		if _, err := l.fp.ReadAt(hdr[:], dataOff-headerSize); err != nil {
			return err
		}
		storedLen := int64(binary.LittleEndian.Uint64(hdr[0:8]))
		newSize = (dataOff - headerSize) + headerSize + storedLen
	}

	if err := l.w.Flush(); err != nil {
		return err
	}
	if err := l.fp.Truncate(newSize); err != nil {
		return err
	}
	if _, err := l.fp.Seek(newSize, io.SeekStart); err != nil {
		return err
	}

	// fix in-memory index
	if l.retainIndex {
		if l.checkpointInterval == 1 {
			if index < uint64(len(l.idx)) {
				l.idx = l.idx[:index]
			}
		} else {
			cpCount := 0
			if index > 0 {
				cpCount = int((index-1)/uint64(l.checkpointInterval) + 1)
			}
			if cpCount < len(l.idx) {
				l.idx = l.idx[:cpCount]
			}
		}
	}
	l.count.Store(index)
	return nil
}

func (l *Log) Sync() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.flushAndSyncLocked()
}

func (l *Log) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if err := l.flushAndSyncLocked(); err != nil {
		_ = l.fp.Close()
		return err
	}

	if l.deleteOnClose {
		_ = l.fp.Close()
		return os.Remove(l.path)
	}

	return l.fp.Close()
}

func (l *Log) flushAndSyncLocked() error {
	if err := l.w.Flush(); err != nil {
		return err
	}
	if l.noSync {
		return nil
	}
	return l.fp.Sync()
}

func (l *Log) locateDataOffsetLocked(index uint64) (int64, error) {
	var hdr [headerSize]byte

	if l.retainIndex {
		if l.checkpointInterval == 1 {
			// full index: direct lookup
			return l.idx[index-1], nil
		}
		// sparse: jump to nearest checkpoint ≤ index, then scan forward
		k := uint64(l.checkpointInterval)
		cp := (index - 1) / k // 0-based checkpoint block

		var hdrOff int64
		var startIdx uint64

		if len(l.idx) > 0 {
			if int(cp) < len(l.idx) {
				hdrOff = l.idx[cp]
				startIdx = cp*k + 1
			} else {
				hdrOff = l.idx[len(l.idx)-1]
				startIdx = (uint64(len(l.idx))-1)*k + 1
			}
		} else {
			hdrOff = 0
			startIdx = 1
		}

		off := hdrOff
		for cur := startIdx; cur < index; cur++ {
			if _, err := l.fp.ReadAt(hdr[:], off); err != nil {
				return 0, err
			}
			storedLen := int64(binary.LittleEndian.Uint64(hdr[0:8]))
			off += headerSize + storedLen
		}
		return off + headerSize, nil
	}

	// no index: linear scan from BOF
	var off int64
	for cur := uint64(1); cur <= index; cur++ {
		if _, err := l.fp.ReadAt(hdr[:], off); err != nil {
			return 0, err
		}
		storedLen := int64(binary.LittleEndian.Uint64(hdr[0:8]))
		if cur == index {
			return off + headerSize, nil
		}
		off += headerSize + storedLen
	}
	return 0, io.EOF
}

func (l *Log) DataOffset(index uint64) (int64, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.locateDataOffsetLocked(index)
}

func (l *Log) ReadAt(p []byte, off int64) (int, error) {
	return l.fp.ReadAt(p, off)
}
