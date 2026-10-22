"""Export Markdown report as a standalone HTML artifact with embedded figures."""
import base64,re
from pathlib import Path
import markdown
root=Path(__file__).resolve().parents[1]
body=markdown.markdown((root/'REPORT.md').read_text(),extensions=['tables','fenced_code'])
def inline_image(m):
 path=root/m.group(1)
 return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"' if path.exists() else m.group(0)
body=re.sub(r'src="(figures/[^\"]+\.png)"',inline_image,body)
css='''body{font:17px/1.6 system-ui,sans-serif;color:#203044;max-width:1080px;margin:48px auto;padding:0 24px}h1{font-size:34px}h2{margin-top:40px;border-bottom:1px solid #d4dce4;padding-bottom:8px}a{color:#125f9b}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px;margin:24px 0}th,td{padding:9px 12px;border-bottom:1px solid #d4dce4;text-align:left}th{background:#edf3f7}code{background:#edf3f7;padding:2px 5px}pre{overflow:auto;padding:16px;background:#edf3f7}pre code{padding:0}@media print{body{font-size:11px;max-width:none;margin:0}h2{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}}'''
(root/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>QuaRot reproduction — SmolLM2-1.7B</title><style>'+css+'</style><body>'+body+'</body></html>')
print('Exported report.html')
