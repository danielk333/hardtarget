

if __name__ == "__main__":

    metadir = Path("/tmp/eiscat/")
    datadir = Path("/tmp/eiscat/uhf/")

    # ts origin
    TS_ORIGIN_SEC_ALIGNED = make_ts_from_str("2024-07-10T12:00:00")
    TS_ORIGIN_SEC_MISALIGNED = make_ts_from_str("2024-07-10T12:03:03")
    TS_ORIGIN_SEC_NOW = dt.datetime.now(dt.timezone.utc).timestamp()
    ts_origin_sec = TS_ORIGIN_SEC_MISALIGNED

    # Sample rate
    SAMPLE_RATE_NUMERATOR = 1000000 
    SAMPLE_RATE_DENOMINATOR = 1
    SAMPLE_BATCH_LENGTH = 12800000  # sample batch length is 12.8 seconds worth of samples



    def write_data():

        # DST
        shutil.rmtree(datadir, ignore_errors=True)
        datadir.mkdir(parents=True, exist_ok=True)

        # writer
        writer = EiscatDRFWriter(
            datadir,
            SAMPLE_RATE_NUMERATOR,
            SAMPLE_RATE_DENOMINATOR * SAMPLE_BATCH_LENGTH,
        )

        # write
        start_idx = writer.index_from_ts(ts_origin_sec)
        n_batches = 36*2
        pointing = get_pointing(n_batches)
        for i in range(n_batches):
            azimuth, elevation = pointing[i]
            writer.write(start_idx + i, azimuth, elevation)





    def write_metadata():

        # DST
        shutil.rmtree(metadir, ignore_errors=True)
        metadir.mkdir(parents=True, exist_ok=True)

        # writer
        writer = EiscatDRFMetadataWriter(
            metadir,
            SAMPLE_RATE_NUMERATOR,
            SAMPLE_RATE_DENOMINATOR * SAMPLE_BATCH_LENGTH,
        )

        # write
        start_idx = writer.index_from_ts(ts_origin_sec)
        n_batches = 36*2
        pointing = get_pointing(n_batches)
        for i in range(n_batches):
            azimuth, elevation = pointing[i]
            writer.write(start_idx + i, azimuth, elevation)


    def read_metadata():

        # read
        reader = EiscatDRFMetadataReader(
            metadir
        )

        (start, end) = reader.get_bounds()
        
        print(start, end, end-start)
        d = reader.read(start, end)
        import pprint

        pprint.pprint(d)

    write()
    read()