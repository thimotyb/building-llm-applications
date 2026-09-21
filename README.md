# Building LLM Applications

Code associated to the book AI Applications with LangChain.

## Python environment and dependencies

Each chapter has its own `requirements.txt` file. Since the required packages and
their versions can differ between chapters, the recommended setup is a separate
virtual environment for each chapter. Create it once, keep it inside the chapter
directory as `.venv`, and reactivate it whenever you return to that chapter; you
do not need to recreate it every time.

For example, to work on chapter 1:

```bash
cd ch01
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

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
