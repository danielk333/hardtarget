
import requests
import argparse
from pathlib import Path
import zipfile
import tempfile
from tqdm import tqdm
import json

try:
    from lxml import html
except ImportError:
    html = None


# Start 
# DAY=20161021 PRODUCT=leo_bpark_2.5u_3P
# url to product page
# https://portal.eiscat.se/schedule/tape2.cgi?exp=leo_bpark_2.5u_3P&date=20161021
# url to download page
# https://portal.eiscat.se/schedule/tape2.cgi?date=20161021&exp=leo_bpark_2.5u_3P&dl=1
# dl == download = true
# location-based download tar or zip format
# https://portal.eiscat.se/schedule/download.cgi?r=249696&r=249698&maxtar=4&filename=leo_bpark_2.5u_3P20161021&submit=Location-based&format=tar
# https://portal.eiscat.se/schedule/download.cgi?r=249696&r=249698&maxtar=4&filename=leo_bpark_2.5u_3P20161021&submit=Location-based&format=zip
# zip download link -- all
# https://rebus.eiscat.se:37009/249696;249698/leo_bpark_2.5u_3P20161021.zip


def get_download_nodes(product_name, day):
    """
    Extract node identifiers from download page
    """
    url = f"https://portal.eiscat.se/schedule/tape2.cgi?exp={product_name}&date={day}&dl=1"
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text

        # parse
        tree = html.fromstring(html_content)
        checkboxes = tree.xpath('//input[@type="checkbox"]')

        def get_item(checkbox):
            return {
                'name': checkbox.get('name'),
                'value': checkbox.get('value'),
            }

        items = [get_item(cb) for cb in checkboxes]
        return [item["value"] for item in items if item["name"] == "r"]
    else:
        print(f"Failed to fetch the HTML content. Status code: {response.status_code}")
        return []


def download(product_name, day, dst, cleanup_zip=True):
    """
    download and extract zip file for (product_name, day) to dst folder.
    cleanup zipfile
    """
    product = f'{product_name}{day}'

    print(f'Started: {product}')

    # check if extracted result already exist
    out = Path(dst) / product
    if out.exists():
        print(f'Completed: {out}')
        return False
    else:
        # create empty directory
        out.mkdir(parents=True, exist_ok=True)
        print(f'Directory created: {out}')

    # check if downloaded zipfile already exists
    temp_directory = Path(tempfile.mkdtemp())
    zip_download = temp_directory / f'{product}.zip'
    if zip_download.exists():
        print(f"Zipfile exists: {zip_download}")
    else:
        # Fetch download nodes
        nodes = get_download_nodes(product_name, day)

        # Download
        zip_url = f'https://rebus.eiscat.se:37009/{";".join(nodes)}/{product}.zip'
        response = requests.get(zip_url, stream=True)
        if response.status_code == 200:
            file_size = int(response.headers.get('Content-Length', 0))
            # Use tqdm to create a progress bar
            print(f'Zipfile: {zip_download}')
            with open(zip_download, 'wb') as file, tqdm(
                desc='Downloading',
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                try:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        bar.update(len(data))
                except KeyboardInterrupt:
                    print('\nDownloading: terminiated')
                    # cleanup
                    zip_download.unlink()
                    out.rmdir()
                    return False
            print(f"Downloading: success: {product}.zip")
        else:
            print(f"Downloading: fail - status code: {response.status_code}")
            return False

    # Extract zip file
    with zipfile.ZipFile(zip_download, 'r') as zip_ref:
        zip_ref.extractall(out)
        print(f'Unzip: {zip_download}')

    # Cleanup zip file
    if zip_download.exists() and cleanup_zip:
        zip_download.unlink()
        print(f'Remove: {zip_download}')

    print(f'Completed; {out}')
    return True


def main():
    """
    Download a list of products from a json file
    """
    parser = argparse.ArgumentParser(description="Download to a destination folder.")
    parser.add_argument('json', help='Json file with products to download')
    parser.add_argument('dst', help='Destination folder for downloaded products')
    args = parser.parse_args()

    with open(args.json, "r") as f:
        json_data = json.load(f)
        for item in json_data:
            download(item["experiment"], item["date"], args.dst)


if __name__ == "__main__":
    main()
