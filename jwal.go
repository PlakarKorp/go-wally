package wally

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"sync"
	"sync/atomic"

	"github.com/golang/snappy"
)

// WAL format:
// File header (little-endian, 16 bytes):
// [0..3]   uint32 magic          // "JWAL" (0x4c41574a)
// [4..5]   uint16 version        // current version = 1
// [6]      uint8  crcAlgo        // checksum algorithm for records (currently only 1=IEEE)
// [7]      uint8  compAlgo       // compression algorithm for records (0=none,1=snappy)
// [8..15]  uint64 reserved

// Record layout (little-endian, 20-byte header):
// [0..7]   uint64 storedLen         // number of bytes stored on disk (after compression if any)
// [8..11]  uint32 crc32(stored)     // checksum of the stored bytes; used for torn-tail recovery
// [12..15] uint32 uncompressedLen   // length of payload before compression; 0 if uncompressed
// [16..19] uint32 reserved
// [20..]   payload (compressed if codec!=none)

const walHdrSize = 16
const recordHdrSize = 20
const dataBase = walHdrSize

const (
	magicJWAL = 0x4c41574a // "JWAL" in LE
	version   = 1

	//crcNone uint8 = 0
	crcIEEE uint8 = 1

	compNone   uint8 = 0
	compSnappy uint8 = 1
)

type fileHdr struct {
	Magic    uint32
	Version  uint16
	CrcAlgo  uint8
	CompAlgo uint8
	Reserved uint64
}

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
	compCodec          uint8

	// counters
	count atomic.Uint64

	// scratch buffers (used under mu)
	compScratch []byte
}

func (l *Log) writeFileHeaderLocked() error {
	var h fileHdr
	h.Magic = magicJWAL
	h.Version = version
	h.CrcAlgo = crcIEEE
	h.CompAlgo = l.compCodec // reuse existing field but now file-level
	// Reserved = 0
	var buf [16]byte
	binary.LittleEndian.PutUint32(buf[0:4], h.Magic)
	binary.LittleEndian.PutUint16(buf[4:6], h.Version)
	buf[6] = h.CrcAlgo
	buf[7] = h.CompAlgo
	// 8..15 zero
	if _, err := l.fp.WriteAt(buf[:], 0); err != nil {
		return err
	}
	return l.fp.Sync()
}

func (l *Log) readFileHeaderLocked() (fileHdr, error) {
	var h fileHdr
	var buf [16]byte
	if _, err := l.fp.ReadAt(buf[:], 0); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			// treat as new empty file; caller will write header
			return h, nil
		}
		return h, err
	}
	h.Magic = binary.LittleEndian.Uint32(buf[0:4])
	h.Version = binary.LittleEndian.Uint16(buf[4:6])
	h.CrcAlgo = buf[6]
	h.CompAlgo = buf[7]
	h.Reserved = binary.LittleEndian.Uint64(buf[8:16])
	return h, nil
}

// Open opens or creates the journal and scans to recover the tail.
func Open(path string, opts *Options) (*Log, error) {
	if opts == nil {
		opts = &Options{}
	}

	var compCodec uint8
	switch opts.Compression {
	case "", "none":
		compCodec = compNone
	case "snappy":
		compCodec = compSnappy
	default:
		fmt.Println("unknown compression codec:", opts.Compression)
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
		compCodec:          compCodec,
	}
	if l.retainIndex {
		if l.checkpointInterval == 0 {
			l.checkpointInterval = 4096
		}
		if l.checkpointInterval < 1 {
			l.checkpointInterval = 1
		}
	}

	hdr, err := l.readFileHeaderLocked()
	if err != nil {
		_ = fp.Close()
		fmt.Println()
		return nil, err
	}

	if hdr.Magic == 0 && hdr.Version == 0 {
		if err := l.writeFileHeaderLocked(); err != nil {
			_ = fp.Close()
			return nil, err
		}
	} else {
		if hdr.Magic != magicJWAL {
			_ = fp.Close()
			return nil, fmt.Errorf("jwal: invalid magic number")
		}
		if hdr.Version != version {
			_ = fp.Close()
			return nil, fmt.Errorf("jwal: unsupported version %d", hdr.Version)
		}
		l.compCodec = hdr.CompAlgo
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
	var off int64 = dataBase // header offset of current record
	var buf []byte
	var hdr [recordHdrSize]byte

	fi, err := l.fp.Stat()
	if err != nil {
		return err
	}
	size := fi.Size()

	if size < dataBase {
		_, err = l.fp.Seek(size, io.SeekEnd)
		return err
	}

	for off+recordHdrSize <= size {
		// read header
		if _, err := l.fp.ReadAt(hdr[:], off); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		storedLen := binary.LittleEndian.Uint64(hdr[0:8])
		end := off + recordHdrSize + int64(storedLen)
		if end > size {
			return l.fp.Truncate(off)
		}
		wantStoredCRC := binary.LittleEndian.Uint32(hdr[8:12])

		// verify checksum over stored bytes (fast, no decompression needed)
		if len(buf) < int(storedLen) {
			buf = make([]byte, int(storedLen))
		}
		if _, err := l.fp.ReadAt(buf[:int(storedLen)], off+recordHdrSize); err != nil {
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
				l.idx = append(l.idx, off+recordHdrSize)
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

	var hdr [recordHdrSize]byte

	hdrOff, err := l.fp.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, err
	}

	// Prepare payload (maybe compressed)
	payload, storedLen, ulen := l.preparePayloadLocked(data)

	// Write header
	binary.LittleEndian.PutUint64(hdr[0:8], storedLen)
	binary.LittleEndian.PutUint32(hdr[8:12], crc32.ChecksumIEEE(payload))
	binary.LittleEndian.PutUint32(hdr[12:16], ulen)
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
			l.idx = append(l.idx, hdrOff+recordHdrSize) // payload start
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

func (l *Log) preparePayloadLocked(data []byte) (payload []byte, storedLen uint64, ulen uint32) {
	switch l.compCodec {
	case compNone:
		return data, uint64(len(data)), 0
	case compSnappy:
		l.compScratch = snappy.Encode(l.compScratch[:0], data)
		return l.compScratch, uint64(len(l.compScratch)), uint32(len(data))
	default:
		return data, uint64(len(data)), 0
	}
}

func (l *Log) Read(index uint64) ([]byte, error) {
	if index == 0 || index > l.count.Load() {
		return nil, io.EOF
	}

	var hdr [recordHdrSize]byte

	dataOff, err := l.locateDataOffsetLocked(index)
	if err != nil {
		return nil, err
	}

	// read header again (dataOff-recordHdrSize)
	if _, err := l.fp.ReadAt(hdr[:], dataOff-recordHdrSize); err != nil {
		return nil, err
	}
	storedLen := int(binary.LittleEndian.Uint64(hdr[0:8]))
	ulen := binary.LittleEndian.Uint32(hdr[12:16])

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
	switch l.compCodec {
	case compNone:
		return buf, nil
	case compSnappy:
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

	var hdr [recordHdrSize]byte

	dataOff, err := l.locateDataOffsetLocked(index)
	if err != nil {
		return nil, err
	}
	if _, err := l.fp.ReadAt(hdr[:], dataOff-recordHdrSize); err != nil {
		return nil, err
	}
	storedLen := int(binary.LittleEndian.Uint64(hdr[0:8]))
	ulen := binary.LittleEndian.Uint32(hdr[12:16])

	// read stored bytes
	tmp := make([]byte, storedLen)
	if _, err := l.fp.ReadAt(tmp, dataOff); err != nil {
		return nil, err
	}
	if crc32.ChecksumIEEE(tmp) != binary.LittleEndian.Uint32(hdr[8:12]) {
		return nil, io.ErrUnexpectedEOF
	}

	switch l.compCodec {
	case compNone:
		// copy into dst if provided; ReadInto contract returns owned slice
		n := len(tmp)
		if cap(dst) < n {
			dst = make([]byte, n)
		} else {
			dst = dst[:n]
		}
		copy(dst, tmp)
		return dst, nil
	case compSnappy:
		// snappy.Decode requires a new buffer; we can allocate exact size if ulen>0
		if ulen > 0 && int(ulen) <= cap(dst) {
			dst = dst[:int(ulen)]
			out, err := snappy.Decode(dst[:0], tmp) // reuse dst backing
			if err != nil {
				return nil, err
			}
			// Sanity-check uncompressed length even on reuse path.
			if ulen != 0 && uint32(len(out)) != ulen {
				return nil, io.ErrUnexpectedEOF
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
func (l *Log) TruncateBack(index uint64) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if index > l.count.Load() {
		return io.EOF
	}

	var hdr [recordHdrSize]byte
	var newSize int64

	if index == 0 {
		newSize = dataBase // keep file header
	} else {
		dataOff, err := l.locateDataOffsetLocked(index)
		if err != nil {
			return err
		}
		if _, err := l.fp.ReadAt(hdr[:], dataOff-recordHdrSize); err != nil {
			return err
		}
		storedLen := int64(binary.LittleEndian.Uint64(hdr[0:8]))
		newSize = (dataOff - recordHdrSize) + recordHdrSize + storedLen
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
	var hdr [recordHdrSize]byte

	if l.retainIndex {
		if l.checkpointInterval == 1 {
			return l.idx[index-1], nil // payload start
		}
		// sparse: start from nearest checkpoint header offset
		k := uint64(l.checkpointInterval)
		cp := (index - 1) / k

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
			hdrOff = dataBase
			startIdx = 1
		}

		off := hdrOff
		for cur := startIdx; cur < index; cur++ {
			if _, err := l.fp.ReadAt(hdr[:], off); err != nil {
				return 0, err
			}
			storedLen := int64(binary.LittleEndian.Uint64(hdr[0:8]))
			off += recordHdrSize + storedLen
		}
		return off + recordHdrSize, nil
	}

	// no index: linear scan from BOF of data region
	off := int64(dataBase)
	for cur := uint64(1); cur <= index; cur++ {
		if _, err := l.fp.ReadAt(hdr[:], off); err != nil {
			return 0, err
		}
		storedLen := int64(binary.LittleEndian.Uint64(hdr[0:8]))
		if cur == index {
			return off + recordHdrSize, nil
		}
		off += recordHdrSize + storedLen
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

// Batch

type Batch struct {
	recs [][]byte
}

func (b *Batch) Add(p []byte) {
	b.recs = append(b.recs, p)
}

func (b *Batch) Reset() {
	b.recs = b.recs[:0]
}

func (b *Batch) Len() int {
	return len(b.recs)
}

func (b *Batch) Records() [][]byte {
	return b.recs
}

func (l *Log) AppendBatch(records ...[]byte) (first, last uint64, err error) {
	if len(records) == 0 {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.count.Load(), l.count.Load(), nil
	}
	return l.appendBatch(records)
}

func (l *Log) WriteBatch(b *Batch) (first, last uint64, err error) {
	if b == nil || len(b.recs) == 0 {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.count.Load(), l.count.Load(), nil
	}
	return l.appendBatch(b.recs)
}

func (l *Log) appendBatch(records [][]byte) (first, last uint64, err error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	var hdr [recordHdrSize]byte

	hdrOff, err := l.fp.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, 0, err
	}

	// Write all headers + payloads
	offs := make([]int64, 0, len(records)) // for full index
	off := hdrOff

	for _, p := range records {
		payload, storedLen, ulen := l.preparePayloadLocked(p)

		// header
		binary.LittleEndian.PutUint64(hdr[0:8], storedLen)
		binary.LittleEndian.PutUint32(hdr[8:12], crc32.ChecksumIEEE(payload))
		binary.LittleEndian.PutUint32(hdr[12:16], ulen)

		if _, err := l.w.Write(hdr[:]); err != nil {
			return 0, 0, err
		}
		if _, err := l.w.Write(payload); err != nil {
			return 0, 0, err
		}

		// advance expected file offset for index math
		offs = append(offs, off+recordHdrSize) // payload start
		off += recordHdrSize + int64(storedLen)
	}

	if err := l.w.Flush(); err != nil {
		return 0, 0, err
	}
	if !l.noSync {
		if err := l.fp.Sync(); err != nil {
			return 0, 0, err
		}
	}

	old := l.count.Load()
	first = old + 1
	nRecs := uint64(len(records))

	if l.retainIndex {
		if l.checkpointInterval == 1 {
			l.idx = append(l.idx, offs...)
		} else {
			// sparse: add checkpoints for records 1, 1+K, ...
			k := uint64(l.checkpointInterval)
			off := hdrOff
			for i := range records {
				next := old + uint64(i) + 1
				if (next-1)%k == 0 {
					l.idx = append(l.idx, off) // header offset
				}
				// recompute storedLen cheaply from the header we wrote:
				// but we already tracked it via offs/advance above:
				// off += headerSize + storedLen
				// We can derive it from consecutive offs:
				if i+1 < len(offs) {
					// next header offset = payloadStartNext - headerSize
					nextHdr := offs[i+1] - recordHdrSize
					off = nextHdr
				} else {
					// final position already in 'off' at end of loop above
				}
			}
		}
	}

	l.count.Add(nRecs)
	last = l.count.Load()
	return first, last, nil
}

// iter

type Iter struct {
	l      *Log
	hdrOff int64  // current header offset
	idx    uint64 // current record index (next to return)
	end    uint64 // inclusive end index snapshot
}

func (l *Log) Iter(from uint64) (*Iter, error) {
	if from == 0 {
		from = 1
	}

	if from > l.count.Load() {
		return nil, io.EOF
	}

	var hdrOff int64
	if from == 1 {
		hdrOff = dataBase
	} else {
		dataOff, err := l.locateDataOffsetLocked(from)
		if err != nil {
			return nil, err
		}
		hdrOff = dataOff - recordHdrSize
	}
	return &Iter{
		l:      l,
		hdrOff: hdrOff,
		idx:    from,
		end:    l.count.Load(), // snapshot
	}, nil
}

func (it *Iter) Next() ([]byte, uint64, error) {
	return it.NextInto(nil)
}

func (it *Iter) NextInto(dst []byte) ([]byte, uint64, error) {
	if it.idx == 0 || it.idx > it.end {
		return nil, 0, io.EOF
	}

	var hdr [recordHdrSize]byte

	l := it.l

	if _, err := l.fp.ReadAt(hdr[:], it.hdrOff); err != nil {
		return nil, 0, err
	}
	storedLen := int(binary.LittleEndian.Uint64(hdr[0:8]))
	wantCRC := binary.LittleEndian.Uint32(hdr[8:12])
	ulen := binary.LittleEndian.Uint32(hdr[12:16])

	dataOff := it.hdrOff + recordHdrSize
	tmp := make([]byte, storedLen)
	if _, err := l.fp.ReadAt(tmp, dataOff); err != nil {
		return nil, 0, err
	}
	if crc32.ChecksumIEEE(tmp) != wantCRC {
		return nil, 0, io.ErrUnexpectedEOF
	}

	var out []byte
	switch l.compCodec {
	case compNone:
		// copy into dst to return an owned slice
		n := len(tmp)
		if cap(dst) < n {
			dst = make([]byte, n)
		} else {
			dst = dst[:n]
		}
		copy(dst, tmp)
		out = dst
	case compSnappy:
		if ulen > 0 && int(ulen) <= cap(dst) {
			dst = dst[:int(ulen)]
			var err error
			out, err = snappy.Decode(dst[:0], tmp) // reuse dst backing
			if err != nil {
				return nil, 0, err
			}
			// Sanity-check uncompressed length on reuse path too.
			if ulen != 0 && uint32(len(out)) != ulen {
				return nil, 0, io.ErrUnexpectedEOF
			}
		} else {
			var err error
			out, err = snappy.Decode(nil, tmp)
			if err != nil {
				return nil, 0, err
			}
			if ulen != 0 && uint32(len(out)) != ulen {
				return nil, 0, io.ErrUnexpectedEOF
			}
		}

	default:
		return nil, 0, io.ErrUnexpectedEOF
	}

	// advance iterator state
	curIdx := it.idx
	it.hdrOff = dataOff + int64(storedLen)
	it.idx++

	return out, curIdx, nil
}
