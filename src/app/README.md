# Uruchomienie projektu

W PowerShell:

```powershell
Set-Location 'C:\Users\przem\Desktop\Licencjat'
.\venv\Scripts\python.exe -m streamlit run app.py
```

W Git Bash:

```bash
cd /c/Users/przem/Desktop/Licencjat
./venv/Scripts/python.exe -m streamlit run app.py
```

Przed pokazaniem końcowych wyników cały pipeline można uruchomić w PowerShell poleceniem:

```powershell
Set-Location 'C:\Users\przem\Desktop\Licencjat'
.\venv\Scripts\python.exe -m src.process_pipeline
```

Pipeline korzysta z obecnych danych lokalnych. Nie pobiera ich ponownie bez flagi `--download-data`.
Trening modeli, FinBERT i generowanie XAI mogą potrwać długo, ale każdy etap zapisuje wynik przed przejściem dalej.
