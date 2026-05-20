import matplotlib.pyplot as plt
from core.volume_data import VolumeData

# ⚠️ CHANGE THIS to point to a small test 3D TIFF on your computer
TEST_FILE_PATH = "mock_droplets_native.tif"

def run_milestone_one_test():
    # 1. Initialize the data object (this triggers loading and MIP math)
    data = VolumeData(TEST_FILE_PATH)

    if data.volume is None:
        print("Test failed: Volume didn't load.")
        return

    # 2. Display the results using Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Milestone 1 Test: {TEST_FILE_PATH}")

    # Show XY (Top-down view)
    axes[0].imshow(data.mip_xy, cmap='gray')
    axes[0].set_title("XY Projection (Top-down)")

    # Show XZ (Side view)
    axes[1].imshow(data.mip_xz, cmap='gray')
    axes[1].set_title("XZ Projection (Side)")

    # Show YZ (Front view)
    axes[2].imshow(data.mip_yz, cmap='gray')
    axes[2].set_title("YZ Projection (Front)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_milestone_one_test()