# GeoHead Adaptation for Few-shot Regression

Geometry-aware head adaptation for few-shot regression, building on
[DARE-GRAM (Nejjar et al., 2023)](https://arxiv.org/abs/2303.13325).

## Research positioning

The goal is **not** to find a representation on which a single shared linear
regressor works for both source and target. Conditional shifts are assumed
to be present, and the source-optimal head and target-optimal head are
generally different. Instead, we want to learn a representation and a
meta-initial head from which **few-shot target labels can rapidly correct
the head toward the target-optimal one**.

DARE-GRAM's inverse-Gram alignment is reinterpreted as a regularizer that
makes such few-shot head correction sample-efficient, rather than as a
mechanism for forcing a shared regressor.

See `docs/design.md` for the full experimental design.

## Setup

### Prerequisites

- pyenv with `pyenv-virtualenv` plugin
- Python 3.11.9 installed via pyenv
- (optional) NVIDIA GPU with CUDA 12.x or 13.x driver

### Environment

We use `pyenv-virtualenv` to manage the environment and `uv` (in `pyproject.toml`
mode) to install dependencies into that env. The `UV_PROJECT_ENVIRONMENT`
variable tells `uv` to install into the pyenv venv instead of creating its own
`.venv`.

```bash
# 1. create the env
pyenv virtualenv 3.11.9 geohead

# 2. activate it for this project
cd /path/to/GeoHead-Adaptation-for-Few-shot-Regression
pyenv local geohead          # writes .python-version

# 3. install deps via uv into the geohead env
pip install --upgrade pip uv
UV_PROJECT_ENVIRONMENT=/home/$USER/.pyenv/versions/geohead \
    uv sync --extra dev --python /home/$USER/.pyenv/versions/3.11.9/bin/python
```

For day-to-day work, `python` and `pip` resolve to the geohead env automatically
because of `.python-version`. Only when running `uv add` / `uv sync` again,
prefix the command with `UV_PROJECT_ENVIRONMENT=$(pyenv prefix)`. A handy alias:

```bash
alias uvg='UV_PROJECT_ENVIRONMENT=$(pyenv prefix) uv'
# then:  uvg add seaborn
```

### Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.11.0+cu128 True
```

## Repository layout

```
docs/                 # design doc and notes
references/           # source papers and discussion notes
src/geohead/          # main package (data, models, losses, training, evaluation)
experiments/          # run configs and scripts
tests/                # unit tests
```
