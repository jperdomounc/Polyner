import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matnifti import (niftiwrite, niftiread, niftiinfo, NiftiInfo,
                      matlab_transform_to_affine, affine_to_matlab_transform)

IN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', 'RANDO_no_implants_1mm', 'LE')
tmp = tempfile.mkdtemp()
fails = []

def check(name, cond, extra=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        fails.append(name)

# ---- 1. byte-for-byte round trip of real MATLAB files -------------------
for fn in ['mask.nii', 'proj_RANDO_Metal_360degrees.nii',
           'fanSensorPosition_fanangle_32f.nii',
           'fanSensorPosition_coneangle_32f.nii']:
    src = os.path.join(IN, fn)
    info = niftiinfo(src)
    data = niftiread(src)
    out = os.path.join(tmp, fn)
    niftiwrite(data, out, info)
    a, b = open(src, 'rb').read(), open(out, 'rb').read()
    if a != b:
        diff = [i for i in range(min(len(a), len(b))) if a[i] != b[i]][:12]
        check('byte-identical: ' + fn, False, f'len {len(a)} vs {len(b)} diffs at {diff}')
    else:
        check('byte-identical: ' + fn, True, f'{data.shape} {data.dtype}')

# ---- 2. no-info path reproduces MATLAB default header ------------------
src = os.path.join(IN, 'mask.nii')
data = niftiread(src)
out = os.path.join(tmp, 'default.nii')
niftiwrite(data, out)   # no info at all
check('default header == MATLAB header',
      open(out, 'rb').read() == open(src, 'rb').read())

# ---- 3. index order matches MATLAB (F-order, no axis reversal) ----------
V = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
p = niftiwrite(V, os.path.join(tmp, 'order'))
check('appends .nii', p.endswith('order.nii'), p)
back = niftiread(p)
check('shape preserved', back.shape == V.shape, str(back.shape))
check('values preserved', np.array_equal(back, V))
raw = open(p, 'rb').read()[352:]
expect = V.tobytes(order='F')
check('data written column-major', raw == expect)

# ---- 4. cross-check against nibabel (which uses MATLAB's convention) ----
try:
    import nibabel as nib
    img = nib.load(p)
    check('nibabel agrees on shape', img.shape == V.shape, str(img.shape))
    check('nibabel agrees on values', np.array_equal(np.asarray(img.dataobj), V))
except ImportError:
    print('SKIP  nibabel cross-check')

# ---- 5. SimpleITK contrast (documents the axis-reversal difference) -----
try:
    import SimpleITK as sitk
    sp = os.path.join(tmp, 'sitk.nii')
    sitk.WriteImage(sitk.GetImageFromArray(V), sp)
    check('sitk reverses axes (expected)',
          niftiread(sp).shape == V.shape[::-1], str(niftiread(sp).shape))
except ImportError:
    print('SKIP  SimpleITK contrast')

# ---- 6. dtypes ----------------------------------------------------------
for dt in ['uint8', 'int16', 'int32', 'int64', 'uint16', 'uint32',
           'uint64', 'int8', 'float32', 'float64']:
    A = (np.arange(24).reshape(2, 3, 4)).astype(dt)
    q = niftiwrite(A, os.path.join(tmp, 'dt_' + dt + '.nii'))
    B = niftiread(q)
    check('dtype ' + dt, B.dtype == A.dtype and np.array_equal(A, B), str(B.dtype))

A = np.zeros((2, 2, 2), bool); A[0, 1, 0] = True
q = niftiwrite(A, os.path.join(tmp, 'logical.nii'))
B = niftiread(q)
check('logical -> uint8', B.dtype == np.uint8 and B[0, 1, 0] == 1
      and niftiinfo(q).Datatype == 'uint8')

# ---- 7. compression -----------------------------------------------------
q = niftiwrite(V, os.path.join(tmp, 'comp'), Compressed=True)
check('Compressed=True -> .nii.gz', q.endswith('comp.nii.gz'), q)
check('gz round trip', np.array_equal(niftiread(q), V))
q2 = niftiwrite(V, os.path.join(tmp, 'comp2.nii.gz'))
check('.gz extension honoured', q2.endswith('.nii.gz') and
      np.array_equal(niftiread(q2), V))

# ---- 8. hdr/img pair ----------------------------------------------------
q = niftiwrite(V, os.path.join(tmp, 'pair'), Combined=False)
check('pair -> .hdr', q.endswith('pair.hdr') and
      os.path.exists(os.path.join(tmp, 'pair.img')))
check('pair round trip', np.array_equal(niftiread(q), V))
check('pair magic ni1', open(q, 'rb').read()[344:348] == b'ni1\x00')
try:
    import nibabel as nib
    check('nibabel reads pair', np.array_equal(
        np.asarray(nib.load(q).dataobj), V))
except ImportError:
    pass

# ---- 9. NIfTI2 ----------------------------------------------------------
q = niftiwrite(V, os.path.join(tmp, 'v2.nii'), Version='NIfTI2')
check('nifti2 sizeof_hdr', open(q, 'rb').read()[:4] == (540).to_bytes(4, 'little'))
check('nifti2 round trip', np.array_equal(niftiread(q), V))
check('nifti2 version reported', niftiinfo(q).Version == 'NIfTI2')
try:
    import nibabel as nib
    n2 = nib.load(q)
    check('nibabel reads NIfTI2', isinstance(n2, nib.Nifti2Image)
          and np.array_equal(np.asarray(n2.dataobj), V))
except ImportError:
    pass

# ---- 10. big endian -----------------------------------------------------
q = niftiwrite(V, os.path.join(tmp, 'be.nii'), Endian='big')
check('big endian round trip', np.array_equal(niftiread(q), V))
check('big endian detected', niftiinfo(q).raw['endian'] == 'big')
try:
    import nibabel as nib
    check('nibabel reads big endian',
          np.array_equal(np.asarray(nib.load(q).dataobj), V))
except ImportError:
    pass

# ---- 11. metadata: spacing, description, units, transforms --------------
info = niftiinfo(os.path.join(IN, 'mask.nii'))
info.PixelDimensions = (0.4, 0.4, 0.4)
info.SpaceUnits = 'Millimeter'
info.Description = 'polyner recon'
info.DisplayIntensityRange = (-1000.0, 3000.0)
d = niftiread(os.path.join(IN, 'mask.nii'))
q = niftiwrite(d, os.path.join(tmp, 'meta.nii'), info)
i2 = niftiinfo(q)
check('pixdim round trip', np.allclose(i2.PixelDimensions, (0.4, 0.4, 0.4)),
      str(i2.PixelDimensions))
check('SpaceUnits round trip', i2.SpaceUnits == 'Millimeter', i2.SpaceUnits)
check('Description round trip', i2.Description == 'polyner recon', repr(i2.Description))
check('DisplayIntensityRange round trip',
      i2.DisplayIntensityRange == (-1000.0, 3000.0), str(i2.DisplayIntensityRange))
try:
    import nibabel as nib
    h = nib.load(q).header
    check('nibabel sees spacing', np.allclose(h['pixdim'][1:4], 0.4),
          str(h['pixdim'][1:4]))
    check('nibabel sees mm units', h.get_xyzt_units()[0] == 'mm',
          str(h.get_xyzt_units()))
except ImportError:
    pass

# sform
A = np.array([[0.4, 0, 0, -40.],
              [0, 0.4, 0, -40.],
              [0, 0, 0.5, -8.],
              [0, 0, 0, 1.]])
info = niftiinfo(os.path.join(IN, 'mask.nii'))
info.TransformName = 'Sform'
info.Transform = affine_to_matlab_transform(A)
q = niftiwrite(d, os.path.join(tmp, 'sform.nii'), info)
i3 = niftiinfo(q)
check('sform round trip', np.allclose(matlab_transform_to_affine(i3.Transform), A))
try:
    import nibabel as nib
    check('nibabel sform matches', np.allclose(nib.load(q).affine, A),
          str(nib.load(q).affine))
except ImportError:
    pass

# qform (rigid: 90deg rotation about z + spacing)
R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
Aq = np.eye(4)
Aq[:3, :3] = R @ np.diag([0.4, 0.4, 0.5])
Aq[:3, 3] = [10., -20., 3.]
info = niftiinfo(os.path.join(IN, 'mask.nii'))
info.TransformName = 'Qform'
info.Transform = affine_to_matlab_transform(Aq)
info.PixelDimensions = (0.4, 0.4, 0.5)
q = niftiwrite(d, os.path.join(tmp, 'qform.nii'), info)
i4 = niftiinfo(q)
check('qform round trip',
      np.allclose(matlab_transform_to_affine(i4.Transform), Aq, atol=1e-5),
      '\n' + str(matlab_transform_to_affine(i4.Transform)))
try:
    import nibabel as nib
    check('nibabel qform matches', np.allclose(nib.load(q).affine, Aq, atol=1e-5),
          '\n' + str(nib.load(q).affine))
except ImportError:
    pass

# qform with a left-handed (negative determinant) affine -> qfac = -1
Aneg = np.diag([-0.4, 0.4, 0.5, 1.0]); Aneg[:3, 3] = [5., 6., 7.]
info = niftiinfo(os.path.join(IN, 'mask.nii'))
info.TransformName = 'Qform'
info.Transform = affine_to_matlab_transform(Aneg)
info.PixelDimensions = (0.4, 0.4, 0.5)
info.Qfactor = -1
q = niftiwrite(d, os.path.join(tmp, 'qneg.nii'), info)
try:
    import nibabel as nib
    check('nibabel qform (qfac=-1) matches',
          np.allclose(nib.load(q).affine, Aneg, atol=1e-5),
          '\n' + str(nib.load(q).affine))
except ImportError:
    pass

# ---- 12. error handling -------------------------------------------------
def expect_err(name, fn):
    try:
        fn(); check(name, False, 'no error raised')
    except Exception as e:
        check(name, True, type(e).__name__ + ': ' + str(e)[:70])

info = niftiinfo(os.path.join(IN, 'mask.nii'))
expect_err('rejects size mismatch',
           lambda: niftiwrite(np.zeros((5, 5, 5), np.float32),
                              os.path.join(tmp, 'bad.nii'), info))
info2 = niftiinfo(os.path.join(IN, 'mask.nii'))
expect_err('rejects dtype mismatch',
           lambda: niftiwrite(np.zeros(info2.ImageSize, np.int16),
                              os.path.join(tmp, 'bad2.nii'), info2))
expect_err('rejects unsupported dtype',
           lambda: niftiwrite(np.zeros((2, 2), np.complex64),
                              os.path.join(tmp, 'bad3.nii')))
expect_err('rejects missing file', lambda: niftiread(os.path.join(tmp, 'nope')))

# truncated file
with open(os.path.join(tmp, 'trunc.nii'), 'wb') as f:
    f.write(open(os.path.join(tmp, 'order.nii'), 'rb').read()[:400])
expect_err('detects truncation', lambda: niftiread(os.path.join(tmp, 'trunc.nii')))

# ---- 13. 2-D / 4-D / 5-D ------------------------------------------------
for shape in [(148,), (1, 148), (64, 64), (10, 11, 12, 3), (4, 5, 6, 2, 3)]:
    A = np.random.rand(*shape).astype(np.float32)
    q = niftiwrite(A, os.path.join(tmp, 'nd.nii'))
    B = niftiread(q)
    check(f'ndim {len(shape)} {shape}', B.shape == shape and np.allclose(A, B),
          str(B.shape))

# ---- 14. non-contiguous / transposed input -----------------------------
A = np.asfortranarray(np.random.rand(6, 7, 8).astype(np.float32))
q = niftiwrite(A, os.path.join(tmp, 'fort.nii'))
check('F-contiguous input', np.array_equal(niftiread(q), A))
A2 = np.random.rand(6, 7, 8).astype(np.float32).transpose(2, 0, 1)
q = niftiwrite(A2, os.path.join(tmp, 'tp.nii'))
check('transposed view input', np.array_equal(niftiread(q), A2))

# ---- 15. scaling --------------------------------------------------------
info = NiftiInfo(ImageSize=(2, 2), PixelDimensions=(1., 1.), Datatype='int16',
                 MultiplicativeScaling=2.0, AdditiveOffset=10.0)
A = np.array([[1, 2], [3, 4]], np.int16)
q = niftiwrite(A, os.path.join(tmp, 'scaled.nii'), info)
check('raw read ignores scaling (MATLAB behaviour)',
      np.array_equal(niftiread(q), A))
check('apply_scaling=True applies it',
      np.allclose(niftiread(q, apply_scaling=True), A * 2.0 + 10.0))

print()
print(('ALL PASS' if not fails else f'{len(fails)} FAILURES: {fails}'))
sys.exit(1 if fails else 0)
