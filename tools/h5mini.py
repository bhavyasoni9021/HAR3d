"""Minimal HDF5 reader (superblock v0, old-style groups, v1 object headers) - enough for Keras weights."""
import struct, numpy as np

class H5:
    def __init__(self, path):
        self.d = open(path, 'rb').read()
        d = self.d
        assert d[:8] == b'\x89HDF\r\n\x1a\n'
        assert d[8] == 0
        self.so, self.sl = d[13], d[14]
        # root symbol table entry begins at 24 + 4*so
        off = 24 + 4*self.so
        _, self.root_oh = struct.unpack_from('<QQ', d, off)[0:2]
        # entry: link name offset(so), object header addr(so), cache type(4), reserved(4), scratch(16)
        self.root_oh = struct.unpack_from('<Q', d, off+8)[0]

    def u(self, fmt, off): return struct.unpack_from('<'+fmt, self.d, off)

    def read_oh(self, addr):
        d = self.d
        ver, _, nmsg, refc, hsize = struct.unpack_from('<BBHII', d, addr)
        assert ver == 1
        msgs = []
        blocks = [(addr+16, hsize)]
        while blocks:
            pos, size = blocks.pop(0)
            end = pos + size
            while pos + 8 <= end:
                t, sz, fl = struct.unpack_from('<HHB', d, pos)
                body = pos + 8
                if t == 0x10:  # continuation
                    o, l = struct.unpack_from('<QQ', d, body)
                    blocks.append((o, l))
                else:
                    msgs.append((t, body, sz))
                pos = body + sz
        return msgs

    def children(self, addr):
        """children of an old-style group: dict name -> object header addr"""
        d = self.d
        msgs = self.read_oh(addr)
        st = [m for m in msgs if m[0] == 0x11]
        if not st: return {}
        btree, heap = struct.unpack_from('<QQ', d, st[0][1])
        assert d[heap:heap+4] == b'HEAP'
        heap_data = struct.unpack_from('<Q', d, heap+24)[0]
        out = {}
        def walk(node):
            assert d[node:node+4] == b'TREE'
            ntype, level, nent = struct.unpack_from('<BBH', d, node+4)
            p = node + 24
            keys_children = []
            p += 8  # key0
            for i in range(nent):
                child = struct.unpack_from('<Q', d, p)[0]; p += 8
                p += 8  # next key
                keys_children.append(child)
            for ch in keys_children:
                if level > 0: walk(ch)
                else:
                    assert d[ch:ch+4] == b'SNOD'
                    ns = struct.unpack_from('<H', d, ch+6)[0]
                    q = ch + 8
                    for _ in range(ns):
                        noff, oh = struct.unpack_from('<QQ', d, q)
                        q += 40
                        e = heap_data + noff
                        name = d[e:d.index(b'\0', e)].decode()
                        out[name] = oh
        walk(btree)
        return out

    def dataset(self, addr):
        d = self.d
        msgs = self.read_oh(addr)
        shape = dtype = layout = None
        for t, body, sz in msgs:
            if t == 0x01:  # dataspace
                ver, rank, flags = struct.unpack_from('<BBB', d, body)
                assert ver == 1
                shape = struct.unpack_from('<%dQ' % rank, d, body+8) if rank else ()
            elif t == 0x03:  # datatype
                cls = d[body] & 0x0f
                size = struct.unpack_from('<I', d, body+4)[0]
                dtype = (cls, size)
            elif t == 0x08:  # layout
                ver = d[body]
                assert ver == 3, ver
                lc = d[body+1]
                if lc == 1:
                    a, s = struct.unpack_from('<QQ', d, body+2); layout = ('contig', a, s)
                elif lc == 0:
                    s = struct.unpack_from('<H', d, body+2)[0]; layout = ('compact', body+4, s)
                else:
                    layout = ('chunked',)
        cls, size = dtype
        npdt = {(1,4):'<f4',(1,8):'<f8',(0,4):'<i4',(0,8):'<i8'}[(cls, size)]
        n = int(np.prod(shape)) if shape else 1
        assert layout[0] in ('contig', 'compact'), layout
        a = layout[1]
        arr = np.frombuffer(d, dtype=npdt, count=n, offset=a).reshape(shape)
        return arr.copy()

    def walk(self, addr=None, prefix=''):
        addr = self.root_oh if addr is None else addr
        for name, oh in self.children(addr).items():
            msgs = self.read_oh(oh)
            if any(m[0] == 0x11 for m in msgs):
                yield from self.walk(oh, prefix + name + '/')
            else:
                yield prefix + name, oh

if __name__ == '__main__':
    import sys
    h = H5(sys.argv[1])
    for p, oh in h.walk():
        try:
            a = h.dataset(oh); print(p, a.shape, a.dtype)
        except Exception as e:
            print(p, 'ERR', e)
