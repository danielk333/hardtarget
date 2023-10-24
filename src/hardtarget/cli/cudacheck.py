#!/usr/bin/env python3

import subprocess
import os


def main():

    dirpath = os.path.dirname(os.path.realpath(__file__))
    srcfile = os.path.join(dirpath, "cudacheck.cu")

    print("Checking Cuda... ")
    subprocess.run(["mkdir", "-p", "/tmp"])
    subprocess.run(["nvcc", srcfile, "-o", "/tmp/hello"])
    result = subprocess.run(["/tmp/hello"], stdout=subprocess.PIPE, text=True)
    subprocess.run(["rm", "/tmp/hello"])
    if result.stdout == "Hello World!\n":
        print("Cuda is running! (and compatible hardware detected)")
    else:
        print("Cuda is NOT running!")


if __name__ == '__main__':
    main()
