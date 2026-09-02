"""MATLAB-compatible NIfTI I/O.

A byte-faithful Python port of MATLAB's ``niftiwrite`` / ``niftiread`` /
``niftiinfo`` (see https://www.mathworks.com/help/images/ref/niftiwrite.html).

Why this exists
---------------
``SimpleITK.GetImageFromArray`` reverses the axis order (numpy axis 0 becomes
the NIfTI *slowest* dimension, i.e. z) and writes an ITK-flavoured header with
``sform_code=1``/``qform_code=1``. MATLAB does neither: it dumps the array in
column-major order straight to disk so that ``V(i,j,k)`` lands at NIfTI voxel
``(i,j,k)``, and leaves the orientation fields zeroed.

Here, ``niftiwrite(V, ...)`` uses ``V.shape`` as ``dim[1..n]`` verbatim and
writes the bytes in Fortran order, so a numpy array round-trips through MATLAB
with the same indexing. No transposes, no flips.

Usage
-----
    from matnifti import niftiwrite, niftiread, niftiinfo

    niftiwrite(vol, 'recon.nii')                       # MATLAB defaults
    niftiwrite(vol, 'recon', Compressed=True)          # -> recon.nii.gz

    info = niftiinfo('recon.nii')
    info.PixelDimensions = (0.4, 0.4, 0.4)
    niftiwrite(vol, 'recon_spaced.nii', info)

    vol = niftiread('recon.nii')                       # raw stored values

Note, as in MATLAB, ``niftiread`` returns the *raw* stored values; it does not
apply ``MultiplicativeScaling``/``AdditiveOffset``. Pass ``apply_scaling=True``
(a Python-only extension) if you want them applied.
"""

from __future__ import annotations

import gzip
import os
import struct
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    'niftiwrite',
    'niftiread',
    'niftiinfo',
    'NiftiInfo',
    'matlab_transform_to_affine',
    'affine_to_matlab_transform',
]


# =============================================
# Constants
# =============================================

# NIfTI datatype code <-> numpy dtype (the subset MATLAB's niftiwrite handles).
_CODE_TO_DTYPE = {
    2: np.dtype('u1'),
    4: np.dtype('i2'),
    8: np.dtype('i4'),
    16: np.dtype('f4'),
    64: np.dtype('f8'),
    256: np.dtype('i1'),
    512: np.dtype('u2'),
    768: np.dtype('u4'),
    1024: np.dtype('i8'),
    1280: np.dtype('u8'),
}
_DTYPE_TO_CODE = {v: k for k, v in _CODE_TO_DTYPE.items()}

# MATLAB Datatype strings are just the class names; 'single' is the odd one out.
_MATLAB_NAME_TO_DTYPE = {
    'single': np.dtype('f4'),
    'double': np.dtype('f8'),
    'int8': np.dtype('i1'),
    'int16': np.dtype('i2'),
    'int32': np.dtype('i4'),
    'int64': np.dtype('i8'),
    'uint8': np.dtype('u1'),
    'uint16': np.dtype('u2'),
    'uint32': np.dtype('u4'),
    'uint64': np.dtype('u8'),
    'logical': np.dtype('u1'),
}
_DTYPE_TO_MATLAB_NAME = {
    np.dtype('f4'): 'single',
    np.dtype('f8'): 'double',
    np.dtype('i1'): 'int8',
    np.dtype('i2'): 'int16',
    np.dtype('i4'): 'int32',
    np.dtype('i8'): 'int64',
    np.dtype('u1'): 'uint8',
    np.dtype('u2'): 'uint16',
    np.dtype('u4'): 'uint32',
    np.dtype('u8'): 'uint64',
}

# xyzt_units is a packed byte: space in bits 0-2, time in bits 3-5.
_SPACE_UNITS = {0: 'Unknown', 1: 'Meter', 2: 'Millimeter', 3: 'Micron'}
_TIME_UNITS = {0: 'None', 8: 'Second', 16: 'Millisecond', 24: 'Microsecond',
               32: 'Hertz', 40: 'PartsPerMillion', 48: 'Radian per second'}
_SPACE_UNITS_INV = {v: k for k, v in _SPACE_UNITS.items()}
_TIME_UNITS_INV = {v: k for k, v in _TIME_UNITS.items()}

_SLICE_CODES = {0: 'Unknown', 1: 'Sequential-Increasing', 2: 'Sequential-Decreasing',
                3: 'Alternating-Increasing', 4: 'Alternating-Decreasing',
                5: 'Alternating-Increasing-2', 6: 'Alternating-Decreasing-2'}
_SLICE_CODES_INV = {v: k for k, v in _SLICE_CODES.items()}

_NIFTI1_HDR_SIZE = 348
_NIFTI1_VOX_OFFSET = 352
_NIFTI2_HDR_SIZE = 540
_NIFTI2_VOX_OFFSET = 544
_NIFTI2_MAGIC_SINGLE = b'n+2\x00\r\n\x1a\n'
_NIFTI2_MAGIC_PAIR = b'ni2\x00\r\n\x1a\n'


# =============================================
# Info struct (MATLAB's niftiinfo return value)
# =============================================

@dataclass
class NiftiInfo:
    """Mirrors the struct MATLAB's ``niftiinfo`` returns.

    ``Transform`` is a 4x4 matrix in MATLAB's ``affine3d`` layout, i.e. the
    *transpose* of the usual NIfTI/nibabel affine (MATLAB post-multiplies row
    vectors). Use :func:`matlab_transform_to_affine` to convert.
    """

    Filename: str = ''
    Filemoddate: str = ''
    Filesize: int = 0
    Description: str = ''
    ImageSize: tuple = ()
    PixelDimensions: tuple = ()
    Datatype: str = 'single'
    BitsPerPixel: int = 32
    SpaceUnits: str = 'Unknown'
    TimeUnits: str = 'None'
    AdditiveOffset: float = 0.0
    MultiplicativeScaling: float = 0.0
    TimeOffset: float = 0.0
    SliceCode: str = 'Unknown'
    FrequencyDimension: int = 0
    PhaseDimension: int = 0
    SpatialDimension: int = 0
    DisplayIntensityRange: tuple = (0.0, 0.0)
    TransformName: str = 'None'
    Transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    Qfactor: int = 1
    Version: str = 'NIfTI1'
    raw: dict = field(default_factory=dict)

    def copy(self) -> 'NiftiInfo':
        import copy as _copy
        return _copy.deepcopy(self)


def _default_info(V: np.ndarray) -> NiftiInfo:
    """The info struct MATLAB synthesises when ``niftiwrite`` gets no ``info``."""
    dtype = _numpy_dtype_for(V)
    return NiftiInfo(
        ImageSize=tuple(V.shape),
        PixelDimensions=(1.0,) * V.ndim,
        Datatype=_DTYPE_TO_MATLAB_NAME[dtype],
        BitsPerPixel=dtype.itemsize * 8,
    )


def _numpy_dtype_for(V: np.ndarray) -> np.dtype:
    if V.dtype == np.bool_:
        return np.dtype('u1')          # MATLAB stores logical as uint8
    if V.dtype not in _DTYPE_TO_MATLAB_NAME:
        raise TypeError(
            f'unsupported array dtype {V.dtype}; niftiwrite handles '
            f'{sorted(_MATLAB_NAME_TO_DTYPE)}'
        )
    return V.dtype


# =============================================
# Transform helpers
# =============================================

def matlab_transform_to_affine(T: np.ndarray) -> np.ndarray:
    """MATLAB ``affine3d.T`` (row-vector convention) -> standard 4x4 affine."""
    return np.asarray(T, dtype=np.float64).T


def affine_to_matlab_transform(A: np.ndarray) -> np.ndarray:
    """Standard 4x4 affine (nibabel/ITK convention) -> MATLAB ``affine3d.T``."""
    return np.asarray(A, dtype=np.float64).T


def _quaternion_from_affine(A: np.ndarray):
    """nifti_mat44_to_quatern: 4x4 affine -> (b, c, d, qoffset, pixdim, qfac)."""
    A = np.asarray(A, dtype=np.float64)
    qoffset = A[:3, 3].copy()
    R = A[:3, :3].astype(np.float64)

    # Column norms give the voxel sizes; strip them to get a rotation.
    d = np.sqrt((R ** 2).sum(axis=0))
    d[d == 0] = 1.0
    R = R / d

    # Polar decomposition via SVD keeps R orthogonal in the face of shear.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    qfac = 1
    if np.linalg.det(R) < 0:
        qfac = -1
        R[:, 2] = -R[:, 2]

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        a = 0.5 * np.sqrt(1.0 + trace)
        b = 0.25 * (R[2, 1] - R[1, 2]) / a
        c = 0.25 * (R[0, 2] - R[2, 0]) / a
        dq = 0.25 * (R[1, 0] - R[0, 1]) / a
    else:                                   # pick the largest diagonal element
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            b = 0.5 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            c = 0.25 * (R[0, 1] + R[1, 0]) / b
            dq = 0.25 * (R[0, 2] + R[2, 0]) / b
            a = 0.25 * (R[2, 1] - R[1, 2]) / b
        elif i == 1:
            c = 0.5 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            b = 0.25 * (R[0, 1] + R[1, 0]) / c
            dq = 0.25 * (R[1, 2] + R[2, 1]) / c
            a = 0.25 * (R[0, 2] - R[2, 0]) / c
        else:
            dq = 0.5 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            b = 0.25 * (R[0, 2] + R[2, 0]) / dq
            c = 0.25 * (R[1, 2] + R[2, 1]) / dq
            a = 0.25 * (R[1, 0] - R[0, 1]) / dq
        if a < 0:
            b, c, dq = -b, -c, -dq

    return float(b), float(c), float(dq), qoffset, d, qfac


def _affine_from_quaternion(b, c, d, qoffset, pixdim, qfac):
    """nifti_quatern_to_mat44."""
    a2 = 1.0 - (b * b + c * c + d * d)
    a = np.sqrt(a2) if a2 > 0 else 0.0
    if a2 <= 0:                              # normalise a degenerate quaternion
        n = np.sqrt(b * b + c * c + d * d)
        if n > 0:
            b, c, d = b / n, c / n, d / n

    R = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
    ], dtype=np.float64)

    scale = np.array([pixdim[0], pixdim[1], pixdim[2] * (qfac if qfac else 1)],
                     dtype=np.float64)
    A = np.eye(4)
    A[:3, :3] = R * scale
    A[:3, 3] = qoffset
    return A


# =============================================
# Header packing
# =============================================

def _blank(s: str, n: int) -> bytes:
    """Char field, blank-padded to n bytes -- MATLAB uses spaces, not NULs."""
    b = s.encode('latin-1', 'replace')[:n]
    return b + b' ' * (n - len(b))


def _pack_nifti1_header(info: NiftiInfo, shape, dtype, endian, single_file, matlab_quirks):
    e = '<' if endian == 'little' else '>'
    hdr = bytearray(_NIFTI1_HDR_SIZE + 4)

    dim = [1] * 8
    dim[0] = len(shape)
    dim[1:len(shape) + 1] = list(shape)

    pixdim = [0.0] * 8
    pixdim[0] = float(info.Qfactor)
    pd = list(info.PixelDimensions)
    pixdim[1:len(pd) + 1] = [float(x) for x in pd]

    qform_code, sform_code, quat, qoff, srow = _orientation_fields(info)

    struct.pack_into(e + 'i', hdr, 0, _NIFTI1_HDR_SIZE)
    hdr[4:14] = _blank('', 10)                      # data_type  (unused)
    hdr[14:32] = _blank('', 18)                     # db_name    (unused)
    struct.pack_into(e + 'i', hdr, 32, 16384 if matlab_quirks else 0)  # extents
    struct.pack_into(e + 'h', hdr, 36, 0)           # session_error
    hdr[38:39] = b'r'                               # regular
    # MATLAB leaves dim_info as a literal space (0x20) rather than packing the
    # freq/phase/slice bitfield. Reproduce that when matlab_quirks is on.
    if matlab_quirks and not (info.FrequencyDimension or info.PhaseDimension
                              or info.SpatialDimension):
        hdr[39:40] = b' '
    else:
        hdr[39:40] = bytes([_pack_dim_info(info)])

    struct.pack_into(e + '8h', hdr, 40, *dim)
    struct.pack_into(e + '3f', hdr, 56, 0.0, 0.0, 0.0)          # intent_p1..3
    struct.pack_into(e + 'h', hdr, 68, 0)                       # intent_code
    struct.pack_into(e + 'h', hdr, 70, _DTYPE_TO_CODE[dtype])
    struct.pack_into(e + 'h', hdr, 72, dtype.itemsize * 8)
    struct.pack_into(e + 'h', hdr, 74, 0)                       # slice_start
    struct.pack_into(e + '8f', hdr, 76, *pixdim)
    struct.pack_into(e + 'f', hdr, 108,
                     float(_NIFTI1_VOX_OFFSET) if single_file else 0.0)
    struct.pack_into(e + 'f', hdr, 112, float(info.MultiplicativeScaling))
    struct.pack_into(e + 'f', hdr, 116, float(info.AdditiveOffset))
    struct.pack_into(e + 'h', hdr, 120, 0)                      # slice_end
    hdr[122] = _SLICE_CODES_INV.get(info.SliceCode, 0)
    hdr[123] = _pack_xyzt_units(info)
    struct.pack_into(e + 'f', hdr, 124, float(info.DisplayIntensityRange[1]))
    struct.pack_into(e + 'f', hdr, 128, float(info.DisplayIntensityRange[0]))
    struct.pack_into(e + 'f', hdr, 132, 0.0)                    # slice_duration
    struct.pack_into(e + 'f', hdr, 136, float(info.TimeOffset))
    struct.pack_into(e + '2i', hdr, 140, 0, 0)                  # glmax, glmin
    hdr[148:228] = _blank(info.Description, 80)
    hdr[228:252] = _blank('', 24)                               # aux_file
    struct.pack_into(e + '2h', hdr, 252, qform_code, sform_code)
    struct.pack_into(e + '3f', hdr, 256, *quat)
    struct.pack_into(e + '3f', hdr, 268, *qoff)
    struct.pack_into(e + '12f', hdr, 280, *srow.ravel())
    hdr[328:344] = _blank('', 16)                               # intent_name
    hdr[344:348] = b'n+1\x00' if single_file else b'ni1\x00'
    return bytes(hdr)


def _pack_nifti2_header(info: NiftiInfo, shape, dtype, endian, single_file, matlab_quirks):
    e = '<' if endian == 'little' else '>'
    hdr = bytearray(_NIFTI2_HDR_SIZE)

    dim = [1] * 8
    dim[0] = len(shape)
    dim[1:len(shape) + 1] = list(shape)

    pixdim = [0.0] * 8
    pixdim[0] = float(info.Qfactor)
    pd = list(info.PixelDimensions)
    pixdim[1:len(pd) + 1] = [float(x) for x in pd]

    qform_code, sform_code, quat, qoff, srow = _orientation_fields(info)

    struct.pack_into(e + 'i', hdr, 0, _NIFTI2_HDR_SIZE)
    hdr[4:12] = _NIFTI2_MAGIC_SINGLE if single_file else _NIFTI2_MAGIC_PAIR
    struct.pack_into(e + 'h', hdr, 12, _DTYPE_TO_CODE[dtype])
    struct.pack_into(e + 'h', hdr, 14, dtype.itemsize * 8)
    struct.pack_into(e + '8q', hdr, 16, *dim)
    struct.pack_into(e + '3d', hdr, 80, 0.0, 0.0, 0.0)          # intent_p1..3
    struct.pack_into(e + '8d', hdr, 104, *pixdim)
    struct.pack_into(e + 'q', hdr, 168,
                     _NIFTI2_VOX_OFFSET if single_file else 0)
    struct.pack_into(e + 'd', hdr, 176, float(info.MultiplicativeScaling))
    struct.pack_into(e + 'd', hdr, 184, float(info.AdditiveOffset))
    struct.pack_into(e + 'd', hdr, 192, float(info.DisplayIntensityRange[1]))
    struct.pack_into(e + 'd', hdr, 200, float(info.DisplayIntensityRange[0]))
    struct.pack_into(e + 'd', hdr, 208, 0.0)                    # slice_duration
    struct.pack_into(e + 'd', hdr, 216, float(info.TimeOffset))
    struct.pack_into(e + '2q', hdr, 224, 0, 0)                  # slice_start/end
    hdr[240:320] = _blank(info.Description, 80)
    hdr[320:344] = _blank('', 24)                               # aux_file
    struct.pack_into(e + '2i', hdr, 344, qform_code, sform_code)
    struct.pack_into(e + '3d', hdr, 352, *quat)
    struct.pack_into(e + '3d', hdr, 376, *qoff)
    struct.pack_into(e + '12d', hdr, 400, *srow.ravel())
    struct.pack_into(e + 'i', hdr, 496, _SLICE_CODES_INV.get(info.SliceCode, 0))
    struct.pack_into(e + 'i', hdr, 500, _pack_xyzt_units(info))
    struct.pack_into(e + 'i', hdr, 504, 0)                      # intent_code
    hdr[508:524] = _blank('', 16)                               # intent_name
    hdr[524] = _pack_dim_info(info)
    hdr[525:540] = b'\x00' * 15                                 # unused_str
    return bytes(hdr) + (b'\x00' * 4 if single_file else b'')


def _pack_dim_info(info: NiftiInfo) -> int:
    return ((int(info.FrequencyDimension) & 0x3)
            | ((int(info.PhaseDimension) & 0x3) << 2)
            | ((int(info.SpatialDimension) & 0x3) << 4))


def _pack_xyzt_units(info: NiftiInfo) -> int:
    space = _SPACE_UNITS_INV.get(info.SpaceUnits)
    time = _TIME_UNITS_INV.get(info.TimeUnits)
    if space is None:
        raise ValueError(f'SpaceUnits must be one of {sorted(_SPACE_UNITS_INV)}')
    if time is None:
        raise ValueError(f'TimeUnits must be one of {sorted(_TIME_UNITS_INV)}')
    return space | time


def _orientation_fields(info: NiftiInfo):
    """-> (qform_code, sform_code, quatern_bcd, qoffset_xyz, srow 3x4)."""
    name = (info.TransformName or 'None').lower()
    zeros3 = (0.0, 0.0, 0.0)
    srow = np.zeros((3, 4))

    if name == 'none':
        return 0, 0, zeros3, zeros3, srow
    if name == 'sform':
        A = matlab_transform_to_affine(info.Transform)
        return 0, 1, zeros3, zeros3, A[:3, :]
    if name == 'qform':
        A = matlab_transform_to_affine(info.Transform)
        b, c, d, qoff, _, _ = _quaternion_from_affine(A)
        return 1, 0, (b, c, d), tuple(qoff), srow
    raise ValueError("TransformName must be 'None', 'Qform' or 'Sform'")


# =============================================
# Header unpacking
# =============================================

def _read_bytes(path: str) -> bytes:
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return f.read()
    with open(path, 'rb') as f:
        return f.read()


def _detect_version(buf: bytes):
    """-> (version, endian). Raises if the buffer is not a NIfTI header."""
    for endian, e in (('little', '<'), ('big', '>')):
        size = struct.unpack_from(e + 'i', buf, 0)[0]
        if size == _NIFTI1_HDR_SIZE:
            return 'NIfTI1', endian
        if size == _NIFTI2_HDR_SIZE:
            return 'NIfTI2', endian
    raise ValueError('not a NIfTI file: sizeof_hdr is neither 348 nor 540')


def _unpack_header(buf: bytes) -> dict:
    version, endian = _detect_version(buf)
    e = '<' if endian == 'little' else '>'
    r = {'version': version, 'endian': endian}

    if version == 'NIfTI1':
        r['dim_info'] = buf[39]
        r['dim'] = struct.unpack_from(e + '8h', buf, 40)
        r['intent_code'] = struct.unpack_from(e + 'h', buf, 68)[0]
        r['datatype'] = struct.unpack_from(e + 'h', buf, 70)[0]
        r['bitpix'] = struct.unpack_from(e + 'h', buf, 72)[0]
        r['pixdim'] = struct.unpack_from(e + '8f', buf, 76)
        r['vox_offset'] = struct.unpack_from(e + 'f', buf, 108)[0]
        r['scl_slope'] = struct.unpack_from(e + 'f', buf, 112)[0]
        r['scl_inter'] = struct.unpack_from(e + 'f', buf, 116)[0]
        r['slice_code'] = buf[122]
        r['xyzt_units'] = buf[123]
        r['cal_max'] = struct.unpack_from(e + 'f', buf, 124)[0]
        r['cal_min'] = struct.unpack_from(e + 'f', buf, 128)[0]
        r['toffset'] = struct.unpack_from(e + 'f', buf, 136)[0]
        r['descrip'] = buf[148:228]
        r['qform_code'] = struct.unpack_from(e + 'h', buf, 252)[0]
        r['sform_code'] = struct.unpack_from(e + 'h', buf, 254)[0]
        r['quatern'] = struct.unpack_from(e + '3f', buf, 256)
        r['qoffset'] = struct.unpack_from(e + '3f', buf, 268)
        r['srow'] = np.array(struct.unpack_from(e + '12f', buf, 280)).reshape(3, 4)
        r['magic'] = buf[344:348]
        r['hdr_size'] = _NIFTI1_HDR_SIZE
        r['default_offset'] = _NIFTI1_VOX_OFFSET
        r['single_file'] = r['magic'].startswith(b'n+1')
    else:
        r['magic'] = buf[4:12]
        r['datatype'] = struct.unpack_from(e + 'h', buf, 12)[0]
        r['bitpix'] = struct.unpack_from(e + 'h', buf, 14)[0]
        r['dim'] = struct.unpack_from(e + '8q', buf, 16)
        r['pixdim'] = struct.unpack_from(e + '8d', buf, 104)
        r['vox_offset'] = struct.unpack_from(e + 'q', buf, 168)[0]
        r['scl_slope'] = struct.unpack_from(e + 'd', buf, 176)[0]
        r['scl_inter'] = struct.unpack_from(e + 'd', buf, 184)[0]
        r['cal_max'] = struct.unpack_from(e + 'd', buf, 192)[0]
        r['cal_min'] = struct.unpack_from(e + 'd', buf, 200)[0]
        r['toffset'] = struct.unpack_from(e + 'd', buf, 216)[0]
        r['descrip'] = buf[240:320]
        r['qform_code'] = struct.unpack_from(e + 'i', buf, 344)[0]
        r['sform_code'] = struct.unpack_from(e + 'i', buf, 348)[0]
        r['quatern'] = struct.unpack_from(e + '3d', buf, 352)
        r['qoffset'] = struct.unpack_from(e + '3d', buf, 376)
        r['srow'] = np.array(struct.unpack_from(e + '12d', buf, 400)).reshape(3, 4)
        r['slice_code'] = struct.unpack_from(e + 'i', buf, 496)[0]
        r['xyzt_units'] = struct.unpack_from(e + 'i', buf, 500)[0]
        r['intent_code'] = struct.unpack_from(e + 'i', buf, 504)[0]
        r['dim_info'] = buf[524]
        r['hdr_size'] = _NIFTI2_HDR_SIZE
        r['default_offset'] = _NIFTI2_VOX_OFFSET
        r['single_file'] = r['magic'].startswith(b'n+2')
    return r


def _info_from_raw(raw: dict, path: str) -> NiftiInfo:
    ndim = max(int(raw['dim'][0]), 1)
    shape = tuple(int(x) for x in raw['dim'][1:ndim + 1])
    pixdim = tuple(float(x) for x in raw['pixdim'][1:ndim + 1])
    qfac = int(raw['pixdim'][0]) if raw['pixdim'][0] in (-1, 1) else 1

    dtype = _CODE_TO_DTYPE.get(raw['datatype'])
    if dtype is None:
        raise ValueError(f"unsupported NIfTI datatype code {raw['datatype']}")

    # dim_info holding a literal space is MATLAB's marker for "unset".
    di = raw['dim_info']
    if di == 0x20:
        di = 0

    if raw['sform_code'] > 0:
        name, affine = 'Sform', np.vstack([raw['srow'], [0, 0, 0, 1]])
    elif raw['qform_code'] > 0:
        b, c, d = raw['quatern']
        name = 'Qform'
        affine = _affine_from_quaternion(b, c, d, raw['qoffset'],
                                         list(raw['pixdim'][1:4]), qfac)
    else:
        name = 'None'
        scale = list(raw['pixdim'][1:4]) + [1.0] * 3
        affine = np.diag([scale[0] or 1.0, scale[1] or 1.0, scale[2] or 1.0, 1.0])

    space = _SPACE_UNITS.get(int(raw['xyzt_units']) & 0x07, 'Unknown')
    time = _TIME_UNITS.get(int(raw['xyzt_units']) & 0x38, 'None')

    try:
        stat = os.stat(path)
        moddate = __import__('datetime').datetime.fromtimestamp(
            stat.st_mtime).strftime('%d-%b-%Y %H:%M:%S')
        size = stat.st_size
    except OSError:
        moddate, size = '', 0

    return NiftiInfo(
        Filename=os.path.abspath(path),
        Filemoddate=moddate,
        Filesize=size,
        Description=raw['descrip'].split(b'\x00')[0].decode('latin-1').rstrip(),
        ImageSize=shape,
        PixelDimensions=pixdim,
        Datatype=_DTYPE_TO_MATLAB_NAME[dtype],
        BitsPerPixel=int(raw['bitpix']),
        SpaceUnits=space,
        TimeUnits=time,
        AdditiveOffset=float(raw['scl_inter']),
        MultiplicativeScaling=float(raw['scl_slope']),
        TimeOffset=float(raw['toffset']),
        SliceCode=_SLICE_CODES.get(int(raw['slice_code']), 'Unknown'),
        FrequencyDimension=di & 0x3,
        PhaseDimension=(di >> 2) & 0x3,
        SpatialDimension=(di >> 4) & 0x3,
        DisplayIntensityRange=(float(raw['cal_min']), float(raw['cal_max'])),
        TransformName=name,
        Transform=affine_to_matlab_transform(affine),
        Qfactor=qfac,
        Version=raw['version'],
        raw=raw,
    )


# =============================================
# Filename resolution
# =============================================

def _resolve_write_paths(filename: str, compressed: bool, combined: bool):
    """-> (header_path, image_path). Equal paths mean a single-file .nii."""
    base = filename
    for ext in ('.gz',):
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            compressed = True
    lower = base.lower()
    if lower.endswith(('.nii', '.hdr', '.img')):
        if lower.endswith('.nii'):
            combined = True
        else:
            combined = False
        base = base[:-4]

    suffix = '.gz' if compressed else ''
    if combined:
        p = base + '.nii' + suffix
        return p, p
    return base + '.hdr' + suffix, base + '.img' + suffix


def _resolve_read_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    for ext in ('.nii', '.nii.gz', '.hdr', '.hdr.gz'):
        if os.path.exists(filename + ext):
            return filename + ext
    raise FileNotFoundError(f'no NIfTI file found for {filename!r}')


def _image_path_for(hdr_path: str) -> str:
    base, gz = hdr_path, ''
    if base.endswith('.gz'):
        base, gz = base[:-3], '.gz'
    if base.lower().endswith('.hdr'):
        base = base[:-4]
    for cand in (base + '.img' + gz, base + '.img', base + '.img.gz'):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f'no .img companion found for {hdr_path!r}')


# =============================================
# Public API
# =============================================

def niftiwrite(V, filename, info=None, *, Compressed=False, Endian='little',
               Version='NIfTI1', Combined=True, matlab_quirks=True):
    """Write ``V`` to a NIfTI file exactly the way MATLAB's ``niftiwrite`` does.

    The array is written in column-major (Fortran) order with ``dim[1..n]`` set
    to ``V.shape``, so ``V[i, j, k]`` in Python is ``V(i+1, j+1, k+1)`` in
    MATLAB after reading the file back. No axis permutation is applied.

    Parameters
    ----------
    V : array_like
        Image data. dtype must be one of MATLAB's supported classes; ``bool``
        is stored as ``uint8``, matching MATLAB's handling of ``logical``.
    filename : str
        Output path. An extension of ``.nii``, ``.nii.gz``, ``.hdr`` or ``.gz``
        is honoured; with no extension, ``.nii`` is appended.
    info : NiftiInfo or dict, optional
        Metadata, normally obtained from :func:`niftiinfo` and edited. As in
        MATLAB, ``info.ImageSize`` and ``info.Datatype`` must agree with ``V``.
    Compressed : bool, default False
        Gzip the output (forces a ``.gz`` extension).
    Endian : {'little', 'big'}, default 'little'
    Version : {'NIfTI1', 'NIfTI2'}, default 'NIfTI1'
    Combined : bool, default True
        ``True`` writes a single ``.nii``; ``False`` writes a ``.hdr``/``.img``
        pair. Ignored when ``filename`` already carries an extension.
    matlab_quirks : bool, default True
        Reproduce MATLAB's cosmetic header quirks (``extents=16384``,
        ``regular='r'``, blank-padded char fields, ``dim_info=0x20``). Set
        ``False`` for a header closer to the reference implementation's.

    Returns
    -------
    str
        The path actually written (the ``.nii``/``.hdr`` file).
    """
    V = np.asarray(V)
    if V.ndim == 0:
        raise ValueError('V must have at least one dimension')
    if V.ndim > 7:
        raise ValueError('NIfTI supports at most 7 dimensions')
    if Endian not in ('little', 'big'):
        raise ValueError("Endian must be 'little' or 'big'")
    if Version not in ('NIfTI1', 'NIfTI2'):
        raise ValueError("Version must be 'NIfTI1' or 'NIfTI2'")

    dtype = _numpy_dtype_for(V)

    if info is None:
        info = _default_info(V)
    else:
        if isinstance(info, dict):
            info = NiftiInfo(**info)
        info = info.copy()
        if tuple(info.ImageSize) and tuple(info.ImageSize) != tuple(V.shape):
            raise ValueError(
                f'info.ImageSize {tuple(info.ImageSize)} does not match the '
                f'size of V {tuple(V.shape)}'
            )
        want = _MATLAB_NAME_TO_DTYPE.get(info.Datatype)
        if want is None:
            raise ValueError(f'unknown info.Datatype {info.Datatype!r}')
        if want != dtype:
            raise ValueError(
                f'info.Datatype is {info.Datatype!r} but V is {V.dtype}; cast V '
                f'or update info.Datatype'
            )
        info.ImageSize = tuple(V.shape)
        if len(info.PixelDimensions) != V.ndim:
            pd = list(info.PixelDimensions)[:V.ndim]
            pd += [1.0] * (V.ndim - len(pd))
            info.PixelDimensions = tuple(pd)
    info.BitsPerPixel = dtype.itemsize * 8

    hdr_path, img_path = _resolve_write_paths(filename, Compressed, Combined)
    single_file = hdr_path == img_path
    compressed = hdr_path.endswith('.gz')

    pack = _pack_nifti1_header if Version == 'NIfTI1' else _pack_nifti2_header
    header = pack(info, V.shape, dtype, Endian, single_file, matlab_quirks)

    out_dtype = dtype.newbyteorder('<' if Endian == 'little' else '>')
    data = np.asarray(V, dtype=dtype).astype(out_dtype, copy=False).tobytes(order='F')

    parent = os.path.dirname(os.path.abspath(hdr_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    opener = (lambda p: gzip.open(p, 'wb')) if compressed else (lambda p: open(p, 'wb'))
    if single_file:
        with opener(hdr_path) as f:
            f.write(header)
            f.write(data)
    else:
        with opener(hdr_path) as f:
            f.write(header)
        with opener(img_path) as f:
            f.write(data)
    return hdr_path


def niftiinfo(filename) -> NiftiInfo:
    """Read the metadata of a NIfTI file, as MATLAB's ``niftiinfo`` does."""
    path = _resolve_read_path(filename)
    buf = _read_bytes(path) if path.endswith('.gz') else open(path, 'rb').read(
        _NIFTI2_HDR_SIZE)
    if len(buf) < _NIFTI1_HDR_SIZE:
        raise ValueError(f'{path!r} is too small to be a NIfTI file')
    return _info_from_raw(_unpack_header(buf), path)


def niftiread(filename, *, apply_scaling=False):
    """Read image data from a NIfTI file, as MATLAB's ``niftiread`` does.

    The returned array has ``shape == info.ImageSize`` and the same voxel
    indexing MATLAB gives, i.e. no axis reversal. Like MATLAB, raw stored
    values are returned; ``apply_scaling=True`` (a Python-only extension)
    applies ``scl_slope``/``scl_inter`` and returns float64.

    ``filename`` may also be a :class:`NiftiInfo`, matching MATLAB's
    ``niftiread(info)`` form.
    """
    if isinstance(filename, NiftiInfo):
        filename = filename.Filename

    hdr_path = _resolve_read_path(filename)
    buf = _read_bytes(hdr_path)
    if len(buf) < _NIFTI1_HDR_SIZE:
        raise ValueError(f'{hdr_path!r} is too small to be a NIfTI file')
    raw = _unpack_header(buf)

    ndim = max(int(raw['dim'][0]), 1)
    shape = tuple(int(x) for x in raw['dim'][1:ndim + 1])
    dtype = _CODE_TO_DTYPE.get(raw['datatype'])
    if dtype is None:
        raise ValueError(f"unsupported NIfTI datatype code {raw['datatype']}")
    dtype = dtype.newbyteorder('<' if raw['endian'] == 'little' else '>')

    if raw['single_file']:
        offset = int(raw['vox_offset']) or raw['default_offset']
        blob = buf
    else:
        offset = int(raw['vox_offset'])
        blob = _read_bytes(_image_path_for(hdr_path))

    count = int(np.prod(shape)) if shape else 0
    need = count * dtype.itemsize
    if len(blob) - offset < need:
        raise ValueError(
            f'{hdr_path!r} truncated: need {need} data bytes at offset {offset}, '
            f'found {len(blob) - offset}'
        )
    data = np.frombuffer(blob, dtype=dtype, count=count, offset=offset)
    data = data.reshape(shape, order='F').astype(dtype.newbyteorder('='))

    if apply_scaling:
        slope = raw['scl_slope']
        if slope not in (0.0,) and np.isfinite(slope):
            data = data.astype(np.float64) * slope + raw['scl_inter']
    return data
