# GESIS Fall Seminar in Computational Social Science 2026
## Introduction to Machine Learning for Text Analysis with Python

### Setup Guide

Welcome! This guide walks you through setting up everything you need for the course. It has three parts:

1. **Local setup** -- a Python environment, the course materials, and an editor on your own laptop
2. **Google Colab** -- a cloud compute service for running notebooks; depending on your hardware, you will probably also need this for the model-training parts of the course (Day 2-3 +)
3. **AI-assisted coding** -- some general tips for using LLMs while learning to code

You are free to deviate from our recommendations if you already have a setup that works for you. Just make sure that, before the course begins, you can run Python code inside a Jupyter notebook. Reach out to us if you encounter issues!

> **⚠️ Work-managed laptops:** If your laptop is managed by your employer, you may lack the permissions to install software. Attempt this setup early, so there is time to involve your ICT administrator if needed. If local installation turns out to be impossible, you can complete the entire course in Google Colab (Part 2).

## Prerequisites

You will need your system terminal to enter commands:

| Operating System | Instructions |
|------------------|--------------|
| **macOS** | Press `⌘ + space`, type `terminal`, press Enter |
| **Windows** | Open the Start menu, type `powershell`, press Enter |

---

# Part 1: Local Setup

## Step 1: Install uv (Python and package management)

We recommend **uv**, a free tool that manages Python versions, environments, and packages in one place. It creates an isolated environment for the course with exactly the package versions we have tested, without touching anything else on your system.

Install it by copy/pasting the following into your terminal:

**macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then close and reopen your terminal, and check that it works:

```bash
uv --version
```

**Additional resources:** https://docs.astral.sh/uv/

## Step 2: Get the course materials

The course materials (exercises, slides, datasets) live in a GitHub repository:

**https://github.com/Rptkiddle/gesis-machine-learning-2026**

> Materials from previous years remain available at the [original course repository](https://github.com/annekroon/gesis-machine-learning).

A repository is a project folder that tracks changes to files over time. We will update the materials during the course week, so you want a copy that can pull in those updates. If it is your first time using git, we recommend using **GitHub Desktop**: https://desktop.github.com/

1. In GitHub Desktop, select **File** → **Clone Repository** → **URL**
2. Enter: `https://github.com/Rptkiddle/gesis-machine-learning-2026.git`
3. Choose a local folder of your choosing

Each morning, press **Fetch origin** / **Pull origin** in GitHub Desktop to sync your copy with ours.


## Step 3: Create the course environment

In your terminal, navigate to the repository folder you just cloned. For example:

```bash
cd ~/Documents/GitHub/gesis-machine-learning-2026
```

(`cd` means "change directory" -- adjust the path to wherever you cloned the repository.)

Then run:

```bash
uv sync
```

This single command installs the right Python version (3.12) and every package the course needs, at the exact versions we have tested. It will take a while and use roughly 3 GB of disk space (the machine learning libraries are large). When it finishes, there will be a hidden `.venv` folder inside the repository containing the environment.

> **Prefer conda or plain venv?** The repository also contains a `requirements.txt` with the same tested package versions. Create and activate an environment with Python 3.12 as you usually would (e.g. `conda create -n gesis_iml python=3.12`, then `conda activate gesis_iml`), and run `pip install -r requirements.txt` inside the repository folder.

## Step 4: Install an editor (IDE) and test everything

Your IDE is where you will spend many long and hopefully enjoyable hours working on your code... so it is important to pick a good one! We recommend **VSCode**: https://code.visualstudio.com/download

> **Note:** If you are on a work-managed laptop and are prompted to choose between a 'user' and a 'system' level setup, choose **user** -- you will hit fewer permissions issues later.

### Install the required extensions

Click the **Extensions** tab on the left-hand side of the interface, then search for and install:

- **Python** -- lets VSCode work with your Python environment(s)
- **Jupyter** -- lets VSCode run Jupyter notebooks (where we will perform the exercises)

### Test your setup

1. In VSCode, select **File** → **Open Folder** and open the course repository folder
2. Open one of the course notebooks, for example `exercises/day0 (prework)/intro_to_python.ipynb`
3. Click **Select Kernel** (top right) → **Python Environments** → choose the one marked `.venv` (this is the environment `uv sync` created) 
4. Run the first code cell (`Shift + Enter`)

If the cell executes and shows output, you are done: you have a working local setup.

> **Note:** You select the kernel once per notebook -- VSCode remembers it afterwards.
>
> **Troubleshoot:** Depending on your system, the environment might have another name, such as "gesis-machine-learning-2026".
> If you don't see a fitting environment listed, trigger a reload via View-menu (top left in VSCode) → **Command Palette** → **Developer: Reload Window**

## A note on hardware

Most of the course runs comfortably on any reasonably recent laptop (say, 8 GB of memory or more). The exception is **Days 3-5**, where we fine-tune a transformer model. This is computationally heavy: without a capable GPU it is slow, and on machines with <16 GB of memory it may fail. During the course, everyone will run the fine-tuning notebooks on Google Colab, which provides a free GPU. So regardless of your hardware, please complete Part 2 as well.

---

# Part 2: Google Colab

[Google Colab](https://colab.research.google.com/) is a free(mium) service that runs Jupyter notebooks in the cloud, on Google's hardware -- including GPUs. Nothing needs to be installed on your laptop; it runs in the browser. We will use it during the course for the computationally heavy exercises.

## Step 1: A Google account

Colab requires a Google account. **We recommend creating a fresh ('burner') Google account for this course** rather than using a personal or institutional one: notebooks and data you open in Colab are processed on Google's servers and associated with the account, and a separate account keeps that cleanly away from your personal data and email. Create one at https://accounts.google.com/signup.

Then log in to https://colab.research.google.com/ with that account.

## Step 2: Open a course notebook

The [course repository README](https://github.com/Rptkiddle/gesis-machine-learning-2026#readme) contains an **"Open in Colab"** badge next to every notebook. Clicking a badge opens that notebook directly in Colab -- no cloning or uploading needed. This is how we will work with Colab during the course.

Note that the notebook that opens is your own private copy. Changes you make are not saved automatically: to keep your work, select **File** → **Save a copy in Drive**. Your copy will then be in your Google Drive under `Colab Notebooks`.

## Step 3: Test it, including the GPU

1. Open any course notebook via its badge (or create an empty notebook at https://colab.research.google.com/)
2. Select **Runtime** → **Change runtime type** → set **Hardware accelerator** to **T4 GPU** → **Save**
3. Add a code cell with the following, and run it (`Shift + Enter`):

```python
import torch
print("GPU available:", torch.cuda.is_available())
```

If the output says `GPU available: True`, Colab is ready.

> **Note:** The free tier of Colab has usage limits. They should be sufficient for the course exercises. Simply close the runtime when you are not using it (**Runtime** → **Disconnect and delete runtime**) to preserve your allowance. Remember to save your work before you disconnect! If you run out of GPU allowance, we recommend purchasing a one-month subscription (~12 euros) as this is slightly more cost-effective than the pay-as-you-go option.

---

# Part 3 (Optional): AI-Assisted Coding

Nothing in this part is required for the course, but AI tools are now part of how nearly everyone writes code, so it is worth setting out how we suggest you use them while learning. We will discuss this further on Day 1.

### A word of realism first

Programming is changing. Increasingly, we describe what we want in natural language and let an AI produce the code. This can be a very productive way of working, and you will probably work this way yourself before long. But it creates a problem for learning: how do you learn how the bricks fit together when a machine will already build you the house? Our advice is to use AI to understand code, use it sparingly to produce code, and avoid using 'agentic' AI (see below) if you are just getting started.

### Chat vs. agentic tools

AI coding tools today come in two forms:

- **Chat:** you ask questions and get answers, in a conversation. The AI never alters your files; you read, judge, and type everything yourself. Examples: any chat interface (Claude, ChatGPT, Gemini), or GitHub Copilot's chat panel in **'Ask' mode**.
- **Agentic:** you describe a goal and the AI acts on it -- editing your files, running commands, fixing its own errors, often making many changes in one go. Examples: Claude Code, Copilot in 'Agent' mode, Cursor.

Agentic tools are powerful, and you are welcome to experiment with them, but (even with experience) they can feel like driving a sports car without knowing where the pedals are. Newer, better models tend to consult the human less (they seem to be less than impressed with our capabilities). Things happen fast, many files change at once, and when something goes wrong you lack the grounding to see what or why.

**For this course, stick to chat.** Which tool you use is up to you -- any of the chat interfaces above works fine, and if you want it inside VSCode, GitHub Copilot's free tier in Ask mode does the job (setup: https://code.visualstudio.com/docs/copilot/setup; Note that approval for the Student or Teacher free tier may take a few days and the application needs to be completed on-site at your institution). Good uses during the course: paste in a piece of exercise code and ask for a line-by-line explanation; paste in an error message and ask what it means; ask why one approach was used rather than another. Used this way, the AI is a patient tutor, and you still largely write and run the code.

> **⚠️ Privacy:** As with all commercial AI offerings, presume that your conversations (and the contents of your code and data) are retained by the provider. Do not paste sensitive data into them. Only a paid account or the Github Copilot Educational (Student or Teacher) free tier allow for an opt-out of data sharing, which you should set manually after activation.

