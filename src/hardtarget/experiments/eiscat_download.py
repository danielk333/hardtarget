import requests
from pathlib import Path
import zipfile
import tempfile
from tqdm import tqdm
from itertools import chain

try:
    from lxml import html
except ImportError:
    html = None


def format_bytes(size):
    """
    Convert bytes to a human-readable format.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0


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


# TODO - should get sizes from html content

def get_download_nodes(day, product_name):
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
        trs = [cb.xpath('./ancestor::tr[1]') for cb in checkboxes]
        flattened_trs = list(chain.from_iterable(trs))

        nodes = []
        for tr in flattened_trs:
            tds = tr.xpath('.//td')
            node = {}
            # tds[0]
            cb = tds[0].find('.//input[@name="r"]')
            node['id'] = cb.get('value')
            # tds[1]
            node['type'] = type = tds[1].text
            # tds[2]
            node['start'] = tds[2].text
            if type == "data":
                # tds[3]
                node['end'] = tds[3].text
                # tds[5]
                # NOTE: assuming number is given in KB
                node['bytes'] = int(tds[5].text) * 1024
            elif type == "info":
                # tds[4]
                tokens = tds[4].text.split(" ")
                node["exp"] = {"type": tokens[1], "name": tokens[2]}
            nodes.append(node)
        return nodes
    else:
        print(f"Failed to fetch the HTML content. Status code: {response.status_code}")
        return []


def download(day, product_name, type, dst, cleanup_zip=True):
    """
    download and extract zip file for (product_name, day) to dst folder.
    cleanup zipfile
    """
    product = f'{day}_{product_name}_{type}'

    # check if extracted result already exist
    out = Path(dst) / product
    if out.exists():
        print(f'Directory exists: {out}')
        return False
    
    
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
        nodes = get_download_nodes(day, product_name)

        # check that exp type is correct
        data_nodes = [node for node in nodes if node["type"] == "data"]
        info_nodes = [node for node in nodes if node["type"] == "info"]
        _type = info_nodes[0]["exp"]["type"]
        if type.casefold() != _type.casefold():
             print(f"Mismatch experiment type, given: {type}, actual {_type}")
             return False

        # Size
        bytes = sum([node["bytes"] for node in data_nodes])
        size = format_bytes(bytes)
        
        # Download
        print(f'Download starting: {product_name} {type} {day} {size}')

        node_ids = [node["id"] for node in nodes]
        zip_url = f'https://rebus.eiscat.se:37009/{";".join(node_ids)}/{product}.zip'
        response = requests.get(zip_url, stream=True)
        if response.status_code == 200:
            file_size = int(response.headers.get('Content-Length', 0)) or bytes
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


if __name__ == "__main__":

    day = "20220408"
    name = "leo_bpark_2.1u_NO"
    type = "uhf"

    # nodes = get_download_nodes(day , name)
    # download(day , name, type, "/tmp")