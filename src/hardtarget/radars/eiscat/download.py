import requests
from pathlib import Path
import zipfile
from tqdm import tqdm
from itertools import chain
import uuid
import shutil
import logging

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

QUERY_URL = "https://portal.eiscat.se/schedule/tape2.cgi"
DOWNLOAD_URL = "https://rebus.eiscat.se:37009"


def get_download_nodes(day, product, logger=None):
    """
    Extract node identifiers from download page
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    url = f"{QUERY_URL}?exp={product}&date={day}&dl=1"
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
        logger.error(f"Failed to fetch the HTML content. Status code: {response.status_code}")
        return []


def download(day, product, type, dst, 
             keep_zip=False, update=False, 
             tmpdir=None, logger=None, progress=False):
    """
    download and extract zip file for (product_name, day, type) to dst folder.
    """
    # logging
    if logger is None:
        logger = logging.getLogger(__name__)

    # destination folders
    data_foldername = f'{product}@{type.lower()}'
    info_foldername = f'{product}@{type.lower()}_information'
    dst = Path(dst)
    if not dst.is_dir():
        logger.warning(f"DST is not a directory {dst}")
        return False, None
    data_dst = dst / data_foldername
    info_dst = dst / info_foldername

    # check if destination folders already exists
    if data_dst.exists() and not update:
        logger.warning(f'Data directory already exists: {data_dst}')
        return False
    if info_dst.exists() and not update:
        logger.warning(f'Info directory already exists: {info_dst}')
        return False, None

    # create temporary directory at destination
    if tmpdir is None:
        tmpdir = Path(dst) / f"{str(uuid.uuid4())}"
    else:
        tmpdir = Path(tmpdir)
    tmpdir.mkdir(exist_ok=True)

    # zip
    zip_filename = f"{product}{day}.zip"
    zip_download = tmpdir / zip_filename
    zip_dst = dst / zip_filename

    # check if downloaded zipfile already exists at dst
    if zip_dst.exists():
        logger.info(f"Zipfile exists: {zip_dst}")
        zip_download = zip_dst
    else:
        # Fetch download nodes
        nodes = get_download_nodes(day, product, logger=logger)
        data_nodes = [node for node in nodes if node["type"] == "data"]
        info_nodes = [node for node in nodes if node["type"] == "info"]

        # check that data matches given type
        _type = info_nodes[0]["exp"]["type"]
        if type.casefold() != _type.casefold():
            logger.warning(f"Mismatch experiment type, given: {type}, actual {_type}")
            return False, None

        # Size
        bytes = sum([node["bytes"] for node in data_nodes])
        size = format_bytes(bytes)

        # Download
        logger.info(f'Product: {product} {day} {size}')

        node_ids = [node["id"] for node in nodes]
        zip_url = f'{DOWNLOAD_URL}/{";".join(node_ids)}/{zip_filename}'
        response = requests.get(zip_url, stream=True)
        completed = False
        if response.status_code == 200:
            file_size = int(response.headers.get('Content-Length', 0)) or bytes
            logger.info(f'Download: zipfile: {zip_download}')
            # Use tqdm to create a progress bar
            pbar = None
            if progress:
                pbar = tqdm(desc="Downloading Eiscat raw data", 
                            total=file_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024
                            )

            with open(zip_download, 'wb') as file:
                try:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        if progress:
                            pbar.update(len(data))
                    completed = True
                    logger.info("Download: completed")
                except KeyboardInterrupt:
                    logger.warning('\nDownload: terminiated')
                    # cleanup
                    shutil.rmtree(tmpdir)
                if progress:
                    pbar.close()        
        else:
            logger.info(f"Download: fail - status code: {response.status_code}")

        if not completed:
            return False, None

    # Extract zip file to random location in dst directory
    with zipfile.ZipFile(zip_download, 'r') as zip_ref:
        logger.info(f'Unzip: {zip_download}')
        zip_ref.extractall(tmpdir)
        logger.info('Unzip: completed')

    # Move or update datafolder and infofolder
    if data_dst.is_dir():
        # update data
        for subdir in (tmpdir / data_foldername).iterdir():
            target_subdir = data_dst / subdir.name
            if not target_subdir.exists():
                shutil.move(subdir, target_subdir) 
    else:
        # mv data
        shutil.move(tmpdir / data_foldername, dst)

    if info_dst.is_dir():
        # update info
        for subdir in (tmpdir / info_foldername).iterdir():
            target_subdir = info_dst / subdir.name
            if not target_subdir.exists():
                shutil.move(subdir, target_subdir) 
    else:
        # mv info
        shutil.move(tmpdir / info_foldername, dst)

    # optionally move zipfile to dst
    if keep_zip:
        if not zip_dst.exists():
            shutil.move(str(zip_download), dst)

    # Cleanup tempdir
    shutil.rmtree(str(tmpdir))

    # Result
    result = {
        "data": str(data_dst), 
        "info": str(info_dst)
    }
    if keep_zip:
        result["zip"] = str(dst / zip_filename)

    logger.info(f"Completed: {str(result)}")
    return True, result


if __name__ == '__main__':

    day = '20220408'
    name = 'leo_bpark_2.1u_NO'
    type = 'uhf'
    logging.basicConfig(level=logging.INFO)
    # nodes = get_download_nodes(day , name)
    download(day, name, type, "/tmp", 
             update=True, keep_zip=True)