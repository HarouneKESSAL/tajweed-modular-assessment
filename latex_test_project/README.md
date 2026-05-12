# LaTeX Test Project

This folder is a small compile-ready test project for the chapter:

- `main.tex`: standalone thesis-style wrapper with the same package style.
- `bibliography.bib`: minimal bibliography file so the project has a complete structure.
- `../sec7_development_experimentation.tex`: the actual chapter included from the repository root.

Compile with XeLaTeX:

```powershell
cd latex_test_project
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

If you move this folder outside the repository, copy `../sec7_development_experimentation.tex`
into this folder and change the `\input{...}` line in `main.tex`.
