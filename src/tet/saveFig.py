import os
import matplotlib.pyplot as plt

def saveFig(fig_id, destination, tight_layout=True, fig_extension="jpg", resolution=300, silent=False):
    """
    Function to save a figure in a specified directory.

    Args:
        fig_id (string): id to save figure under
        destination (string): Directory to save figure into
        tight_layout (bool, optional): Make use of tight layout. Defaults to True.
        fig_extension (str, optional): File extension for saved figure. Defaults to "jpg".
        resolution (int, optional): Resolution of saved figure. Defaults to 300.
    """
    PROJECT_ROOT_DIR = destination
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
    os.makedirs(IMAGES_PATH, exist_ok=True)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    if not silent:
        print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)