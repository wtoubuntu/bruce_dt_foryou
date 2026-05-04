# 📘 Professional Git Workflow Tutorial

This guide outlines the "Shared Repository" workflow used for Bruce's Data Viz tool. Use this as a reference whenever you start a new feature or clean up your work.

---

## 1. Starting a New Feature
Always start from a fresh, updated `main` branch.

### Step A: Sync your computer
```bash
git checkout main
git pull origin main
```

### Step B: Create and switch to a new branch
```bash
git checkout -b feature-name-here
```

### Step C: Connect your branch to GitHub
This only needs to be done **once** per branch.
```bash
git push -u origin feature-name-here
```

---

## 2. Saving Your Work
As you write code, save your progress frequently.

### Step A: Stage your changes
```bash
git add filename.py
# OR to add everything:
git add .
```

### Step B: Create a commit (Save point)
```bash
git commit -m "Brief description of what you changed"
```

### Step C: Push to GitHub
Since you used `-u` earlier, you can now just type:
```bash
git push
```

---

## 3. Merging Into Main
When your feature is finished and tested, move it into the master `main` branch.

### Step A: Move to Main
```bash
git checkout main
git pull origin main
```

### Step B: The Merge Command
```bash
git merge feature-name-here
```

### Step C: "The Vim Trap" (How to escape the text screen)
If a strange text screen with `~` symbols appears, do NOT panic.
1. Press `Esc`
2. Type `:wq`
3. Press `Enter`

### Step D: Update the Cloud
```bash
git push origin main
```

---

## 4. Cleaning Up
Keep your environment tidy by deleting branches you no longer need.

### Delete the Local Branch (Your laptop)
```bash
git branch -d feature-name-here
```

### Delete the Remote Branch (GitHub website)
```bash
git push --delete origin feature-name-here
```

---

## ⚠️ Pro-Tips
- **Fatal Credential Cache Warning:** If you see "fatal: credential-cache unavailable", you can safely ignore it. This is normal on Windows.
- **Check Status:** If you are confused, type `git status`. It will tell you where you are and what needs to be saved.