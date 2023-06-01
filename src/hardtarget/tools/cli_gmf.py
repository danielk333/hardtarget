#!/usr/bin/env python

"""
CLI for basic GMF function.
"""

import argparse

def process(config_file, src, dst=None):
    # Perform file processing logic here
    print("Config file:", config_file)
    print("Source path:", src)
    if dst is None:
        print("Print to screen")
    else:
        print("Destination path", dst)

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="File processing script")

    # Add the arguments
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("src", help="Path to source directory")
    parser.add_argument("--dst", help="Path to destination directory", default=".")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to process the file
    process(args.config, args.src, args.dst)


if __name__ == "__main__":
    main()