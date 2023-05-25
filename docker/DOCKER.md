## Build Hardtarget Image

```bash
docker build -t htimage -f docker/Dockerfile .
```

## Run Hardtarget Image

```bash
docker run --rm -it --gpus all  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 htimage 
```
