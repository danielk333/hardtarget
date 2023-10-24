import digital_rf as drf
import pprint

target = "/home/AD.NORCERESEARCH.NO/inar/Data/hard_target/leo_bpark_2.1u_NO@uhf/20210412_11"
reader = drf.DigitalRFReader(target)

channels = reader.get_channels()
for chnl in channels:
    print(f"CHANNEL: {chnl}")
    props = reader.get_properties(chnl)
    print("props:")
    pprint.pprint(props)
    bounds = list(reader.get_bounds(chnl))
    print(f"bounds: {bounds}")
    blocks = reader.get_continuous_blocks(bounds[0], bounds[1], chnl)
    print(f"blocks: {blocks}")
    data = reader.read_vector(bounds[0], 10, chnl)
    print(f"data: {data}")
