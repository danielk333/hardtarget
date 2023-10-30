import h5py
from astropy import units as u


def write(file, data):
    with h5py.File(file, "w") as file:
        # Store the data in a dataset
        dataset = file.create_dataset("data", data=data.value)

        print(data_with_units.unit.to_string())

        # Store the unit as an attribute or in a separate dataset
        dataset.attrs["unit"] = data_with_units.unit.to_string()


def read(file):

    with h5py.File(file, "r") as file:
        dataset = file["data"]

        # Retrieve the data values
        data_values = dataset[:]

        # Retrieve the unit information
        unit_str = dataset.attrs["unit"]
        data_with_units = data_values * u.Unit(unit_str)

        return data_with_units


if __name__ == __main__:

    FILE = "data_with_units.h5"
    data_with_units = [1.0, 2.0, 3.0] * u.meter
    write(FILE)
    data = read(FILE)
    print(data)