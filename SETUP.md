# Setup (Windows, VS Code, GitHub)

Do these once, in order. Total time: about 30 minutes, most of it the data download.

## 1. Make the repo and put these files in it

1. On github.com, create a new **public** repository called `beat-the-grade`. Tick "Add a README".
2. Open PowerShell and run:

   ```powershell
   cd $HOME\Projects
   git clone https://github.com/YOUR-USERNAME/beat-the-grade.git
   cd beat-the-grade
   ```

3. Unzip the starter files and copy everything inside into this `beat-the-grade` folder. Let it replace the README.

## 2. Create the Python environment

Still in PowerShell, inside the folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

You'll know it worked when your prompt starts with `(.venv)`.

## 3. Get the data

1. Sign in to kaggle.com and search for the dataset **"Lending Club"** by **wordsforthewise** (loans 2007 to 2018Q4).
2. Click **Download**. It's a large zip.
3. Unzip it and put the contents in the `data` folder of this project. The notebook searches inside `data` for the file named `accepted_2007_to_2018Q4.csv` (or `.csv.gz`), so nested folders are fine.

## 4. Your Gemini key (needed later, for the stereotype test)

1. Get a free key at Google AI Studio.
2. Copy `.env.example`, rename the copy to `.env`, and paste the key after `GEMINI_API_KEY=`.

`.env` is listed in `.gitignore`, so it never reaches GitHub.

## 5. Open it in VS Code and run the first notebook

```powershell
code .
```

1. Install the **Python** and **Jupyter** extensions if VS Code asks.
2. Open `01_data.ipynb`.
3. Top right, click **Select Kernel** and choose the `.venv` environment.
4. Click **Run All**. Loading the big file takes a few minutes.

## 6. Save your progress to GitHub

Do this every couple of hours, and always with the notebook outputs visible (judges read them):

```powershell
git add .
git commit -m "Describe what you did"
git push
```

The data folder is ignored, so only your code and results get pushed.
