import os
import requests
import zipfile

target_dir = os.path.join("data", "dataset", "redial", "nltk")
os.makedirs(target_dir, exist_ok=True)
zip_path = os.path.join(target_dir, "redial_nltk.zip")

# Remove any old corrupted 8KB HTML stub
if os.path.exists(zip_path):
    size = os.path.getsize(zip_path)
    if size < 1024 * 1024:
        print(f"Removing invalid file ({size} bytes)...")
        os.remove(zip_path)

# Live mirrors
mirrors = [
    # 1. Original SharePoint URL with custom session headers
    "https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EdVnNcteOkpAkLdNL-ejvAABPieUd8jIty3r1jcdJvGLzw?download=1",
    # 2. Backup GitHub raw mirror
    "https://raw.githubusercontent.com/RUCAIBox/CRSLab/master/data/redial_nltk.zip"
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

downloaded = False
for url in mirrors:
    print(f"\nAttempting download from: {url}")
    try:
        session = requests.Session()
        r = session.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
        if r.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(zip_path)
            print(f"Downloaded file size: {file_size / (1024 * 1024):.2f} MB")
            
            # Check if this is a real zip archive (> 5MB)
            if zipfile.is_zipfile(zip_path) and file_size > 5 * 1024 * 1024:
                print("Valid zip archive confirmed!")
                downloaded = True
                break
            else:
                print("Downloaded file is not a valid zip archive (likely an HTML login page).")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
    except Exception as e:
        print(f"Failed with error: {e}")

if downloaded:
    print(f"\nExtracting archive into {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Files successfully extracted! Folder contents:")
    for item in os.listdir(target_dir):
        print(" ->", item)
else:
    print("\nCould not automatically stream the zip file due to SharePoint access limitations.")
    print(f"You can also paste the URL directly into your Chrome/Edge browser:")
    print("https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EdVnNcteOkpAkLdNL-ejvAABPieUd8jIty3r1jcdJvGLzw?download=1")
    print(f"Once downloaded, move 'redial_nltk.zip' into: {os.path.abspath(target_dir)}")