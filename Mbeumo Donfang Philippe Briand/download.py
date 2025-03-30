import os
import hashlib
from icrawler.builtin import GoogleImageCrawler

# Filepath to your labels.csv
LABELS_FILE = r"labels.csv"

# Directory to save images
SAVE_DIR = r"./images"

def get_categories_from_file(labels_file):
    """
    Extract category names from a labels file (e.g., CSV or TXT).
    """
    categories = []
    with open(labels_file, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            class_id, category = line.strip().split(",", 1)
            categories.append(category)
    return categories

def calculate_image_hash(image_path):
    """
    Calculate the hash of an image file to detect duplicates.
    """
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def download_images_for_category(category, category_index, num_images=10):
    """
    Download images for a specific category using icrawler.
    """
    category_dir = os.path.join(SAVE_DIR, str(category_index))
    os.makedirs(category_dir, exist_ok=True)

    # Check existing images
    existing_images = os.listdir(category_dir)
    existing_hashes = {calculate_image_hash(os.path.join(category_dir, img)) for img in existing_images}
    num_existing_images = len(existing_images)

    if num_existing_images >= num_images:
        print(f"Category {category} (Index: {category_index}) already has {num_existing_images} images. Skipping...")
        return

    # Calculate the number of images to download
    images_to_download = num_images - num_existing_images
    print(f"Category {category} (Index: {category_index}) needs {images_to_download} more images.")

    # Use GoogleImageCrawler to fetch images
    google_crawler = GoogleImageCrawler(storage={'root_dir': category_dir})
    google_crawler.crawl(
        keyword=category,
        max_num=images_to_download,
        file_idx_offset=num_existing_images
    )

    # Rename images to follow the naming convention and avoid duplicates
    downloaded_images = os.listdir(category_dir)
    for i, image_name in enumerate(downloaded_images):
        old_path = os.path.join(category_dir, image_name)
        new_name = f"{category_index}_image_{i + 1}.jpg"
        new_path = os.path.join(category_dir, new_name)

        # Check for duplicate images
        image_hash = calculate_image_hash(old_path)
        if image_hash in existing_hashes:
            os.remove(old_path)  # Remove duplicate
            print(f"Duplicate image detected and removed: {old_path}")
        else:
            if os.path.exists(new_path):
                print(f"File {new_path} already exists. Skipping renaming for {old_path}.")
                continue  # Skip renaming if the target file already exists
            os.rename(old_path, new_path)
            existing_hashes.add(image_hash)

def main():
    # Get categories from the labels file
    categories = get_categories_from_file(LABELS_FILE)

    # Download images for each category
    for index, category in enumerate(categories):
        print(f"Processing category: {category} (Index: {index})")
        download_images_for_category(category, index)

if __name__ == "__main__":
    main()