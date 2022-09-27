
### How to access `lmp` module within files under lmp_practice?

1. Add in language-model-playground level `.vscode/settings.json`
    Write in:
    ```
    {
    "python.analysis.extraPaths": [
        "/Users/yangqingwen/Desktop/Github/language-model-playground",]
    }
    ```
2. Add in Preferences/User Settings (JSON):
```    "jupyter.notebookFileRoot": "${workspaceFolder}",```
3. Add `__init__.py` in `lmp_practice` directory.
4. Try reloading the window (Cmd+Shift+P, type: Reload Windows).
5. Open a python file under `lmp_practice` and try `import lmp`.
