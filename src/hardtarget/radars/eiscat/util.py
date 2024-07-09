PARBL_ELEVATION = 8
PARBL_AZIMUTH = 9
PARBL_END_TIME = 10
PARBL_SEQUENCE = 11
PARBL_START_TIME = 42


def parse_matlab(mat, sample_rate, file_secs):

    """
    parse (and check) matlab file
    sample_rate - expected sample rate
    file_secs - expected file length in seconds
    returns dictionary with meta information
    """
    d = {}

    parbl = mat["d_parbl"][0]

    # global start of recording - repeated for all files
    d["t_origin"] = t_origin = float(parbl[PARBL_START_TIME])

    # end time of last sample in this file (where the last sample ends)
    d["t_end"] = t_end = float(parbl[PARBL_END_TIME])

    # number of data samples in file
    d["samples"] = samples = len(mat["d_raw"])
    
    # file duration in seconds from number of actual samples
    d["duration"] = duration = float(file_secs)

    # check that number of samples match expected file size
    _duration = float(samples / sample_rate)
    assert duration == _duration , f"mismatch file duration: expected {duration}s, actual {_duration}s"

    # sequence number
    # files have a sequence number relative to t0, with 1 as the first logical sequence number.
    seq = int(parbl[PARBL_SEQUENCE])

    # check that sequence number match timestamps 
    _seq = round((t_end - t_origin) / duration)
    assert seq == _seq, f"mismatch sequence number {seq} with timestamps {_seq}" 

    # file index
    # subtract one to get a zero-indexed file index from sequence numbers
    d["idx_file"] = idx_file = seq - 1
        
    # global index of first sample in recording (t_origin)
    d["idx_origin"] = idx_origin = round(t_origin * sample_rate)

    # global index of first sample in file
    d["idx_first"] = idx_first = idx_origin + idx_file * samples

    # timestamp of first sample in file
    d["t_start"] = t_end - duration

    # global index of last sample in file
    d["idx_last"] = idx_first + samples - 1    

    # instrument pointing angles
    d["elevation"] = elevation = float(parbl[PARBL_ELEVATION])
    d["azimuth"] = azimuth = float(parbl[PARBL_AZIMUTH])

    return d
