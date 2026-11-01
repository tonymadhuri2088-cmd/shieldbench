# One-click: runs every untouched test set, judges, reports, writes the datasheet and runs the regression gate.
# Usage (PowerShell, inside the project folder, venv active):   .\run_all.ps1            (add -Suites heldout4 for just one)
param([string]$Suites = "all", [string]$Target = "ollama")
python -m shieldbench.autorun --dry-run --suites $Suites --target $Target
$ans = Read-Host "Run this plan now? It uses your Ollama bot and a little Claude credit. Do NOT press Ctrl+C once it starts. (y/n)"
if ($ans -eq "y") { python -m shieldbench.autorun --suites $Suites --target $Target }
