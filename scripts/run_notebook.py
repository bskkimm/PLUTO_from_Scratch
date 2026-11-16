"""Execute the offline notebook code in this fresh Python process."""
import json
from pathlib import Path

notebook = json.loads(Path('notebooks/pluto_e2e_walkthrough.ipynb').read_text())
namespace = {'__name__': '__notebook__'}
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        if cell['execution_count'] is not None or cell['outputs']:
            raise ValueError('notebook must not contain saved execution outputs')
        exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
print('Notebook executed in a fresh process.')
