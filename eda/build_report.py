"""Put the analysis, the tests and the findings into the page: eda/out/report.json + inference.json + features.json + model.json + findings.py +
template.html -> eda/out/report.html.

    python eda/build_report.py
"""
import json
from pathlib import Path

from findings import FEATURES, INFERENCE, MODELS, INFERENCE_OVERVIEW, OVERVIEW, TABLES

EDA = Path(__file__).resolve().parent
report = json.loads((EDA / 'out' / 'report.json').read_text(encoding='utf-8'))
inference = json.loads((EDA / 'out' / 'inference.json').read_text(encoding='utf-8'))
for section in report['sections']:
    section['findings'] = TABLES[section['id']]
    section['inference'] = {**inference[section['id']], 'findings': INFERENCE[section['id']]}
report['cross']['findings'] = TABLES['across']
report['cross']['inference'] = {**inference['across'], 'findings': INFERENCE['across']}
report['tests'] = inference['summary']
report['models'] = {**json.loads((EDA / 'out' / 'model.json').read_text(encoding='utf-8')), 'findings': MODELS}
report['features'] = {**json.loads((EDA / 'out' / 'features.json').read_text(encoding='utf-8')), 'findings': FEATURES}
report['overview'] = {**OVERVIEW, 'inference': INFERENCE_OVERVIEW}

data = json.dumps(report, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
html = (EDA / 'template.html').read_text(encoding='utf-8').replace('/*DATA*/', data)
out = EDA / 'out' / 'report.html'
out.write_text(html, encoding='utf-8')
print(f'{out} {out.stat().st_size / 1e6:.2f} MB')
