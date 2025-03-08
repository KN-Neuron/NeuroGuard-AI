# AI

W projekcie uzywamy Pythona 3.12.7

## Instrukcja Poetry

1. W terminalu odpalcie `python3 -m venv venv`, albo `py -m venv venv` jeśli jesteście na Windowsie.

2. Aktywujcie venva: `source venv/bin/activate`, lub `.\venv\Scripts\activate`.

3. Zainstalujcie Poetry: `pip install poetry`.

4. Zainstalujcie wszystkie libki: `poetry install`.

5. Jeśli chcecie dodać jakieś dependency: `poetry add <nazwa>`, np. `poetry add requests`.

6. Po instalacji zróbcie sobie `pip install -e .` w folderze, gdzie jest setup.py i powinno działać.
