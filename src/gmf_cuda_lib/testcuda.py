#!/usr/bin/env python3
import subprocess

print("Testing Cuda. This may take a few seconds... ")
subprocess.run(["mkdir", "-p", "/tmp"])
subprocess.run(["nvcc", "hello.cu", "-o", "/tmp/hello"])
result = subprocess.run(["/tmp/hello"], stdout=subprocess.PIPE, text=True)
subprocess.run(["rm", "/tmp/hello"])

if result.stdout == "Hello Hello \n":
    print("Cuda is running!")
else:
    print("Cuda is not running!")

