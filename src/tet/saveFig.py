import os
import matplotlib.pyplot as plt

def saveFig(fig_id, destination, tight_layout=True, fig_extension="jpg", resolution=300):
    # Where to save the figures
    PROJECT_ROOT_DIR = destination
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
    os.makedirs(IMAGES_PATH, exist_ok=True)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)