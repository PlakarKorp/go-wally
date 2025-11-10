package jwal

import (
	"encoding/binary"
	"hash/crc32"
	"io"

	"github.com/golang/snappy"
)

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
			out, err = snappy.Decode(dst[:0], tmp)
			if err != nil {
				return nil, 0, err
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
