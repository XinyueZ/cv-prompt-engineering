## Install via Docker

```bash

# setup
docker build --no-cache --tag cv-prompt-engineering -f Dockerfile .

# run
docker run --gpus all -v /home/ubuntu/work/cv-prompt-engineering/:/workspace/    -p 5555:5555 --rm  -it --shm-size=55gb -d cv-prompt-engineering tail -f /dev/null

```

## Run

```bash
streamlit run app.py --server.port 5555 --server.enableCORS false
```
