import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from .dt_standard import dt_format, str_to_dt


def search_for_sentinel_data(
    dt_start: dt.datetime,
    dt_end: dt.datetime,
    collection: str = "SENTINEL-2",
    object: str = "S2B",
    product_catalogue: str = "AUX_POEORB",
):
    """
    Search for data from a specific object between certain times

    https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html#sentinel-2-precise-orbit-determination-pod-products

    TODO: Add to only search for S2B, S2A does not have the correct orbit.

    Args:
        dt_start: start time of file
        dt_end: end time of file
        collection: collection to search available data from {SENTINEL-1/SENTINEL-2/SENTINEL-3}
        object: Specific satellite to get data from within the collection
        product_catalogue: what product catalogue to search from {"AUX_GNSSRD" (RINEX) / "AUX_PROQUA" (Quaternions) / "AUX_POEORB" (Orbit)}
    Returns:
        returns a list of data from the object from the specific product catalogue with a start time within dt_start and dt_end
    """

    query = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=((Collection/Name eq '{collection}') and (ContentDate/Start gt {dt_start.strftime(dt_format)}Z) and (ContentDate/Start lt {dt_end.strftime(dt_format)}Z) and ((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name eq 'productType' and i0/Value eq '{product_catalogue}'))))&$orderby=ContentDate/Start&$top=10"
    json_res = requests.get(query).json()

    value = json_res["value"]

    if not value:
        raise Exception(f"No data for {collection} found between: {dt_start} and {dt_end}")

    return value


def get_orbit_data_id(dt_start: dt.datetime, dt_end: dt.datetime):
    """
    Args:
        dt_start: datetime
        dt_end: datetime
    """

    # orbit data files are 24h long, with start at 22:59-23:59, so for any input we take back the search one day
    data = search_for_sentinel_data(dt_start - dt.timedelta(days=1), dt_end, "SENTINEL-2", "AUX_POEORB")

    # TODO check each file and which has closest start date
    # for i,value in enumerate(data):
    #       if value[ContentDate][Start] - dt_start < prev_delta
    #            best_id = i

    # TODO: Once the query only searches for S2B this 1 can be changed to 0
    return data[1]["Id"]


def extract_eof_data_block(data_path: Path, dt_start: dt.datetime, dt_end: dt.datetime):
    """

    Returns:
        list of tuples, each tuple is the timepoint with a list of x,y,z,vx.vy,vz

    """

    tree = ET.parse(str(data_path))
    root = tree.getroot()
    data_block = root.find("Data_Block")

    def data_structure(x: ET.Element):
        return [
            x.find("X").text,
            x.find("Y").text,
            x.find("Z").text,
            x.find("VX").text,
            x.find("VY").text,
            x.find("VZ").text,
        ]

    data = [
        (x.find("UTC").text[4:], data_structure(x))
        for x in data_block.findall(".//OSV")
        if time_within_block(x.find("UTC").text[4:], dt_start, dt_end)
    ]

    if not data:
        raise Exception(
            f"No data available between {dt_start.strftime(dt_format)} and {dt_end.strftime(dt_format)}, try a larger timespan"
        )

    return data


def time_within_block(dt_curr: str | dt.datetime, dt_start: dt.datetime, dt_end: dt.datetime) -> bool:

    if isinstance(dt_curr, str):
        dt_curr = str_to_dt(dt_curr)
    return dt_curr >= dt_start and dt_curr <= dt_end


def generate_token(username: str, password: str) -> str:

    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }

    response = requests.post(
        url,
        headers=headers,
        data=data,
    ).json()

    if "access_token" in response:
        return response["access_token"]
    else:
        raise Exception("Bad credentials, no access token generated")


def download_orbit_data(data_id: str, access_token: str, output_dir: Path) -> Path:
    """
    https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Satellite_constellation

    Args:
        dt_start: start of data
        dt_end: end of data
    Returns:
        Orbit data within dt_start and dt_end format (timepoint, [x y z vx vy vz] (m, m/s))
    """
    data_path = output_dir / "data.eof"
    if data_path.is_file():
        return data_path

    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({data_id})/$value"

    headers = {"Authorization": f"Bearer {access_token}"}

    # Create a session and update headers
    session = requests.Session()
    session.headers.update(headers)

    # Perform the GET request
    response = session.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(str(data_path), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        return data_path
    else:
        print(f"GET response: {response.text}")
        raise Exception(f"Failed to download orbit data. Status code: {response.status_code} ")
