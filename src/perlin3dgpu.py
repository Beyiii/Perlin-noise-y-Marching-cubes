import cupy as cp

def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def generate_perlin_noise_3d(shape, res, tileable=(False, False, False), interpolant=interpolant):
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    
    grid = cp.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1], 0:res[2]:delta[2]]
    grid = cp.transpose(grid, (1, 2, 3, 0)) % 1

    theta = 2 * cp.pi * cp.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    phi = 2 * cp.pi * cp.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = cp.stack(
        (cp.sin(phi) * cp.cos(theta),
         cp.sin(phi) * cp.sin(theta),
         cp.cos(phi)),
        axis=3
    )

    if tileable[0]:
        gradients[-1,:,:] = gradients[0,:,:]
    if tileable[1]:
        gradients[:,-1,:] = gradients[:,0,:]
    if tileable[2]:
        gradients[:,:,-1] = gradients[:,:,0]

    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)

    g000 = gradients[:-d[0], :-d[1], :-d[2]]
    g100 = gradients[d[0]:, :-d[1], :-d[2]]
    g010 = gradients[:-d[0], d[1]:, :-d[2]]
    g110 = gradients[d[0]:, d[1]:, :-d[2]]
    g001 = gradients[:-d[0], :-d[1], d[2]:]
    g101 = gradients[d[0]:, :-d[1], d[2]:]
    g011 = gradients[:-d[0], d[1]:, d[2]:]
    g111 = gradients[d[0]:, d[1]:, d[2]:]

    # Ramps
    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    z = grid[:, :, :, 2]
    stack = cp.stack

    n000 = cp.sum(stack((x, y, z), axis=3) * g000, axis=3)
    n100 = cp.sum(stack((x - 1, y, z), axis=3) * g100, axis=3)
    n010 = cp.sum(stack((x, y - 1, z), axis=3) * g010, axis=3)
    n110 = cp.sum(stack((x - 1, y - 1, z), axis=3) * g110, axis=3)
    n001 = cp.sum(stack((x, y, z - 1), axis=3) * g001, axis=3)
    n101 = cp.sum(stack((x - 1, y, z - 1), axis=3) * g101, axis=3)
    n011 = cp.sum(stack((x, y - 1, z - 1), axis=3) * g011, axis=3)
    n111 = cp.sum(stack((x - 1, y - 1, z - 1), axis=3) * g111, axis=3)

    # Interpolation
    t = interpolant(grid)
    tx, ty, tz = t[:, :, :, 0], t[:, :, :, 1], t[:, :, :, 2]

    n00 = n000 * (1 - tx) + tx * n100
    n10 = n010 * (1 - tx) + tx * n110
    n01 = n001 * (1 - tx) + tx * n101
    n11 = n011 * (1 - tx) + tx * n111

    n0 = n00 * (1 - ty) + ty * n10
    n1 = n01 * (1 - ty) + ty * n11

    return (1 - tz) * n0 + tz * n1
