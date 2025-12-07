import os


categories = [
    "cat",
    "basketball",
    "bird",
    "sword",
    "ice cream",
    "mug",
    "fish",
    "stop sign",
]


def download_quickdraw_data(data_dir, categories):
    """
    Download Google Quick Draw dataset numpy files
    Download from: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
    """
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

    print("Downloading Google Quick Draw dataset...")
    for category in categories:
        filename = f"{category}.npy"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            url = f"{base_url}/{filename.replace(' ', '%20')}"
            print(f"Downloading {category}...")
            urllib.request.urlretrieve(url, filepath)

    print("Download complete!")


if __name__ == "__main__":
    data_dir = "./quick_draw_data"
    download_quickdraw_data(data_dir, categories)
