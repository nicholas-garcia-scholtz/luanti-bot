"""Decompression helpers.

Luanti compresses map blocks and definition packets with zlib (older
serialisation formats) or zstd (serialisation format >= 29). Python 3.14+
ships zstd in the stdlib (`compression.zstd`); on older versions libzstd is
bound directly with ctypes -- either way the bot stays dependency-free.
"""
import ctypes
import ctypes.util
import zlib

try:
    from compression.zstd import ZstdDecompressor
except ImportError:
    ZstdDecompressor = None

_lib = None


def _zstd():
    global _lib
    if _lib is None:
        path = (ctypes.util.find_library("zstd")
                or ctypes.util.find_library("libzstd")   # Windows: libzstd.dll
                or "libzstd.so.1")
        _lib = ctypes.CDLL(path)
        _lib.ZSTD_createDStream.restype = ctypes.c_void_p
        _lib.ZSTD_freeDStream.argtypes = [ctypes.c_void_p]
        _lib.ZSTD_initDStream.argtypes = [ctypes.c_void_p]
        _lib.ZSTD_initDStream.restype = ctypes.c_size_t
        _lib.ZSTD_decompressStream.argtypes = [ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_void_p]
        _lib.ZSTD_decompressStream.restype = ctypes.c_size_t
        _lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        _lib.ZSTD_isError.restype = ctypes.c_uint
        _lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        _lib.ZSTD_getErrorName.restype = ctypes.c_char_p
    return _lib


class _Buf(ctypes.Structure):
    _fields_ = [("dst", ctypes.c_void_p),
                ("size", ctypes.c_size_t),
                ("pos", ctypes.c_size_t)]


def zstd_decompress_stream(data: bytes, out_chunk: int = 1 << 16):
    """Decompress one zstd frame from the front of `data`.

    Luanti embeds compressed data inline in a larger packet with no length
    prefix, so this returns (plaintext, bytes_consumed) and stops at the end of
    the first frame.
    """
    if ZstdDecompressor is not None:
        d = ZstdDecompressor()
        out = d.decompress(data)
        if not d.eof:
            raise RuntimeError("zstd: truncated frame")
        return out, len(data) - len(d.unused_data)

    lib = _zstd()
    ds = lib.ZSTD_createDStream()
    if not ds:
        raise RuntimeError("ZSTD_createDStream failed")
    try:
        rc = lib.ZSTD_initDStream(ds)
        if lib.ZSTD_isError(rc):
            raise RuntimeError(lib.ZSTD_getErrorName(rc).decode())

        src_raw = ctypes.create_string_buffer(data, len(data))
        src = _Buf(ctypes.cast(src_raw, ctypes.c_void_p), len(data), 0)
        out = bytearray()
        dst_raw = ctypes.create_string_buffer(out_chunk)
        while True:
            dst = _Buf(ctypes.cast(dst_raw, ctypes.c_void_p), out_chunk, 0)
            rc = lib.ZSTD_decompressStream(ds, ctypes.byref(dst), ctypes.byref(src))
            if lib.ZSTD_isError(rc):
                raise RuntimeError("zstd: " + lib.ZSTD_getErrorName(rc).decode())
            out += dst_raw.raw[:dst.pos]
            if rc == 0:                      # frame complete
                break
            if dst.pos == 0 and src.pos >= src.size:
                raise RuntimeError("zstd: truncated frame")
        return bytes(out), src.pos
    finally:
        lib.ZSTD_freeDStream(ds)


def zlib_decompress_stream(data: bytes):
    """Same contract as above, for zlib-compressed payloads."""
    d = zlib.decompressobj()
    out = d.decompress(data)
    return out, len(data) - len(d.unused_data)


def decompress_stream(data: bytes, ser_ver: int):
    """Pick the codec Luanti uses for the given serialisation version."""
    if ser_ver >= 29:
        return zstd_decompress_stream(data)
    return zlib_decompress_stream(data)
