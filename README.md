# Prompt Engineering for Computer Vision (experimental)

<div>
<img src="./assets/basic.gif"   />
</div>

## Description

The repo is used for studying how to use Prompt Engineering for Computer Vision tasks. 

- Use state-of-art models like `diffusion` or other baseline models to generate, inpaint, and paint images.
```bash
streamlit run basic_app.py --server.port 5555 --server.enableCORS false
```
- To-be-continued....

#### Working comments

I personally think that prompting is a new programming approach. Don’t assume that guiding models with natural language is easy. On the contrary, I believe it’s quite the opposite. Natural language programming lacks the syntax of traditional programming languages, which means there are no type checks or any protective mechanisms in place. If the model (AI) receives an inappropriate prompt, the generated results can be completely different from what was expected.

Here is a prompt I have used the diffusion model in computer vision. Although it has brought some surprises, it is not actually my ultimate goal.

[Blog](https://teetracker.medium.com/ai-new-trend-prompt-engineering-3d7369dcbd86)

## Install via Docker

```bash

# setup
docker build --no-cache --tag cv-prompt-engineering -f Dockerfile .

# run
docker run --gpus all -v /home/ubuntu/work/cv-prompt-engineering/:/workspace/    -p 5555:5555 --rm  -it --shm-size=55gb -d cv-prompt-engineering tail -f /dev/null

```

## Run

```bash
streamlit run basic_app.py --server.port 5555 --server.enableCORS false
```
