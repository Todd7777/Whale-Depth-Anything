# Push to Whale-Depth-Anything (one-time, from your Mac)

Commit is prepared locally. GitHub auth must run on **your** machine:

```bash
cd /path/to/Whale-Depth-Anything   # or Depth-Anything clone

git remote add whale https://github.com/Todd7777/Whale-Depth-Anything.git 2>/dev/null || true

git push -u whale main
```

If HTTPS asks for credentials, use a [Personal Access Token](https://github.com/settings/tokens) as the password, or switch to SSH:

```bash
git remote set-url whale git@github.com:Todd7777/Whale-Depth-Anything.git
git push -u whale main
```

**Optional:** On GitHub, set the repo description to *CETI / AVATARS underwater depth on Depth Anything* and pin **README_WHALE_DEPTH.md** (or rename it to `README.md` on GitHub for the landing page).
