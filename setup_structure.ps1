# À exécuter depuis PowerShell, dans le dossier du projet :
# cd "C:\Users\GABUNTU\OneDrive - USherbrooke\Bureau\projet_de_vie\prototype scoring veltro"
# .\setup_structure.ps1

# --- 1. Création des dossiers ---
$dossiers = @("data\raw", "data\processed", "notebooks", "src", "models", "tests", "reports")
foreach ($d in $dossiers) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Write-Host "Dossiers crees." -ForegroundColor Green

# --- 2. Deplacement des fichiers existants vers leur bon emplacement ---
# Ajuste les noms si besoin selon ce que tu as reellement dans ton dossier actuel

if (Test-Path "veltro_synthetic_deals.csv") {
    Move-Item "veltro_synthetic_deals.csv" "data\raw\" -Force
    Write-Host "veltro_synthetic_deals.csv -> data\raw\" -ForegroundColor Cyan
}

if (Test-Path "Eda.ipynb") {
    Move-Item "Eda.ipynb" "notebooks\" -Force
    Write-Host "Eda.ipynb -> notebooks\" -ForegroundColor Cyan
}

if (Test-Path "generate_syntetic_data.py") {
    Move-Item "generate_syntetic_data.py" "src\" -Force
    Write-Host "generate_syntetic_data.py -> src\" -ForegroundColor Cyan
}

if (Test-Path "canevas_eda.pdf") {
    Move-Item "canevas_eda.pdf" "reports\" -Force
    Write-Host "canevas_eda.pdf -> reports\" -ForegroundColor Cyan
}

Write-Host "`nStructure prete. Verifie avec: tree /F" -ForegroundColor Green
