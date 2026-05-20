import torch
import numpy as np
import random
import tifffile
import os

# --- Configurations from LD_Counting.ipynb ---
VOLUME_SHAPE = (64, 256, 256)
MAX_CLUSTER_COUNT = 6
MIN_CLUSTER_COUNT = 1
MAX_LDS_CLASS =  {
    "small": 45,
    "medium": 12,
    "large": 2,
    'Xlarge':1
}
CLASSES = {
    "small": (1, 3),
    "medium": (4, 8),
    "large": (9, 20),
    'Xlarge':(50,90)
}

# --- Core Generation Logic from Notebook ---
def generate_cluster_centers(shape, min_clusters=MIN_CLUSTER_COUNT, max_clusters=MAX_CLUSTER_COUNT,
                             margin=(10, 30, 30), deterministic=False, seed=42,
                             force_distribution=None,max_dist=(98,300,300)):
    if deterministic:
        random.seed(seed)

    z, y, x = shape
    mz, my, mx = margin
    num_clusters = random.randint(min_clusters, max_clusters)
    clusters = []
    attempts = 0

    while len(clusters) < num_clusters and attempts < 10 * num_clusters:
        cz = random.randint(mz, z - mz)
        cy = random.randint(my, y - my)
        cx = random.randint(mx, x - mx)

        spread = (
            random.randint(50, max_dist[0]),
            random.randint(50,  max_dist[1]),
            random.randint(50,  max_dist[2])
        )

        spread_dist = ((spread[0]**2 + spread[1]**2 + spread[2]**2) ** 0.5) * 0.5

        too_close = False
        for existing in clusters:
            ez, ey, ex = existing["center"]
            d = ((cz - ez) ** 2 + (cy - ey) ** 2 + (cx - ex) ** 2) ** 0.5
            if d < spread_dist:
                too_close = True
                break

        if not too_close:
            dist_type = force_distribution if force_distribution else random.choice(['normal', 'uniform'])
            ld_counts = {
                "Xlarge": random.randint(0, MAX_LDS_CLASS["Xlarge"]),
                "large": random.randint(0, MAX_LDS_CLASS["large"]),
                "small": random.randint(1, MAX_LDS_CLASS["small"]),
                "medium": random.randint(1, MAX_LDS_CLASS["medium"]),
            }

            clusters.append({
                "center": (cz, cy, cx),
                "spread": spread,
                "distribution": dist_type,
                "ld_counts": ld_counts
            })

        attempts += 1
    return clusters

def draw_gaussian_blob(radius, intensity_range=(0.6, 1.0), crop_factor=3):
    rz = ry = rx = float(radius) if isinstance(radius, (int, float)) else float(radius[0])
    
    dz = int(np.ceil(2 * crop_factor * rz)) | 1
    dy = int(np.ceil(2 * crop_factor * ry)) | 1
    dx = int(np.ceil(2 * crop_factor * rx)) | 1

    z = torch.arange(dz) - dz // 2
    y = torch.arange(dy) - dy // 2
    x = torch.arange(dx) - dx // 2
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    blob = torch.exp(-((zz**2) / (2 * rz**2) +
                       (yy**2) / (2 * ry**2) +
                       (xx**2) / (2 * rx**2)))

    peak = random.uniform(*intensity_range)
    radius_px=(dz/2,dy/2,dx/2)
    return blob * peak, radius_px

def draw_ellipsoid_mask(radius, intensity_range=(0.6,1.0), crop_factor=3):
    rz = ry = rx = float(radius) if isinstance(radius, (int, float)) else float(radius[0])

    dz = int(np.ceil(2 * crop_factor * rz)) | 1
    dy = int(np.ceil(2 * crop_factor * ry)) | 1
    dx = int(np.ceil(2 * crop_factor * rx)) | 1

    z = torch.arange(dz) - dz // 2
    y = torch.arange(dy) - dy // 2
    x = torch.arange(dx) - dx // 2
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    mask = torch.exp(-((zz**2) / (2 * rz**2) +
                       (yy**2) / (2 * ry**2) +
                       (xx**2) / (2 * rx**2))) > 0.01

    radius_px = (dz / 2, dy / 2, dx / 2)
    peak = random.uniform(*intensity_range)
    return mask.to(torch.float32)*peak, radius_px

def draw_blob(radius, tag='ellipsoid', intensity_range=(0.6, 1.0), crop_factor=3):
    if tag == 'gaussian': return draw_gaussian_blob(radius, intensity_range, crop_factor)
    return draw_ellipsoid_mask(radius, intensity_range, crop_factor)

def try_place_blob(volume, mask, center, radius, intensity_range=(0.6, 1.0), threshold=0.01, blob_type='ellipsoid'):
    blob, rad_px = draw_blob(radius, blob_type, intensity_range)
    bz, by, bx = blob.shape
    cz, cy, cx = center
    Z, Y, X = volume.shape
    rz, ry, rx = bz // 2, by // 2, bx // 2

    kz0, kz1 = -min(cz-rz,0), bz-max(cz+rz-Z+1,0)
    ky0, ky1 = -min(cy-ry,0), by-max(cy+ry-Y+1,0)
    kx0, kx1 = -min(cx-rx,0), bz-max(cx+rx-X+1,0)

    vz0, vz1 = max(cz-rz,0), min(cz+rz+1,Z)
    vy0, vy1 = max(cy-ry,0), min(cy+ry+1,Y)
    vx0, vx1 = max(cx-rx,0), min(cx+rx+1,X)

    blob_crop = blob[kz0:kz1, ky0:ky1, kx0:kx1]
    mask_crop = mask[vz0:vz1, vy0:vy1, vx0:vx1]

    if blob_crop.shape != mask_crop.shape: return False, volume, mask, None
    if (mask_crop & (blob_crop > threshold)).any(): return False, volume, mask, None

    volume[vz0:vz1, vy0:vy1, vx0:vx1] += blob_crop
    mask[vz0:vz1, vy0:vy1, vx0:vx1] |= (blob_crop > threshold)

    meta = {"center": (cz, cy, cx), "radius": radius, "bbox": ((vz0, vz1), (vy0, vy1), (vx0, vx1))}
    return True, volume, mask, meta

def generate_ld_volume(volume_shape, blob_types=['gaussian','ellipsoid']):
    Z, Y, X = volume_shape
    volume = torch.zeros((Z, Y, X), dtype=torch.float32)
    mask = torch.zeros_like(volume, dtype=torch.bool)
    metadata = []
    
    selected_blob_type = random.choice(blob_types)
    print(f"Generating Volume using blob type: {selected_blob_type}")
    
    clusters = generate_cluster_centers(shape=volume_shape)

    for cluster_id, cluster in enumerate(clusters):
        cz, cy, cx = cluster["center"]
        sz, sy, sx = cluster["spread"]
        dist_mode = cluster["distribution"]
        ld_counts = cluster["ld_counts"]

        for cls_name, r_range in CLASSES.items():
            num_ld = ld_counts[cls_name]
            for _ in range(num_ld):
                for _ in range(3):
                    if dist_mode == 'normal':
                        dz = int(np.random.normal(0, sz * 0.25))
                        dy = int(np.random.normal(0, sy * 0.25))
                        dx = int(np.random.normal(0, sx * 0.25))
                    else:
                        dz = random.randint(-sz // 2, sz // 2)
                        dy = random.randint(-sy // 2, sy // 2)
                        dx = random.randint(-sx // 2, sx // 2)

                    z = min(max(cz + dz, 5), Z - 5)
                    y = min(max(cy + dy, 5), Y - 5)
                    x = min(max(cx + dx, 5), X - 5)
                    r = random.uniform(*r_range)

                    success, volume, mask, meta = try_place_blob(volume, mask, (z, y, x), r, blob_type=selected_blob_type)
                    if success:
                        meta.update({"class": cls_name, "cluster_id": cluster_id})
                        metadata.append(meta)
                        break
                        
    return volume, mask, metadata

def create_mock_data():
    print("Initializing mock generation via notebook code...")
    vol_tensor, mask_tensor, metadata = generate_ld_volume(VOLUME_SHAPE)
    
    # Add background noise to make it realistic 
    print("Adding background noise...")
    noise = torch.randn(VOLUME_SHAPE) * 0.05
    vol_tensor += noise
    
    # Scale to 8-bit for efficient viewing
    vol_np = vol_tensor.numpy()
    vol_np = np.clip((vol_np / vol_np.max()) * 255, 0, 255).astype(np.uint8)
    
    filename = "mock_droplets_native.tif"
    filepath = os.path.join(os.getcwd(), filename)
    tifffile.imwrite(filepath, vol_np)
    
    print(f"Native mock data saved to: {filepath}")
    print(f"Total droplets placed: {len(metadata)}")

if __name__ == "__main__":
    create_mock_data()