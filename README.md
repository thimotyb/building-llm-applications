# Building LLM Applications

Code associated to the book AI Applications with LangChain.

## Python version

The requirements target LangChain 1.x, which needs **Python 3.10 to 3.13**
(3.12 is recommended). On Python 3.8 or 3.9, pip fails with errors such as
`No matching distribution found for langchain==1.0.3`. Python 3.14 resolves,
but some pinned packages have no prebuilt wheels for it yet.

Check your version with `python3 --version`. If it is too old, the simplest
cross-platform option is [uv](https://docs.astral.sh/uv/), which installs a
Python 3.12 next to the system one without replacing it:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12
```

Alternatively, install Python 3.12 from python.org (Windows/macOS), Homebrew
(`brew install python@3.12`), or your distribution's packages. On Linux, do not
replace or relink the system `python3`: other system tools depend on it.

## Python environment and dependencies

Each chapter has its own `requirements.txt` file. Since the required packages and
their versions can differ between chapters, the recommended setup is a separate
virtual environment for each chapter. Create it once, keep it inside the chapter
directory as `.venv`, and reactivate it whenever you return to that chapter; you
do not need to recreate it every time.

For example, to work on chapter 1:

```bash
cd ch01
uv venv --python 3.12 --seed .venv   # or: python3.12 -m venv .venv
source .venv/bin/activate
python --version                     # should print 3.12.x
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`--seed` installs pip inside the environment, so the usual `pip` commands work.
If your system `python3` is already 3.10–3.13, plain `python3 -m venv .venv`
works too.

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

When you finish working, deactivate it with:

```bash
deactivate
```

When returning to the same chapter, only activation is necessary:

```bash
cd ch01
source .venv/bin/activate
```

Repeat the initial setup in another chapter only the first time you use it. A
single virtual environment at the repository root may be convenient, but it can
accumulate unused packages or version conflicts as you move between chapters,
so it is not the recommended default for this repository.
