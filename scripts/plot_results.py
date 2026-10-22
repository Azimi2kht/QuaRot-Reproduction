"""Generate seminar figures and measured summaries; never fill missing results."""
import json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
NAMES=['baseline','rotation_only','unrotated_4bit','quarot_4bit']
LABELS={'baseline':'FP16','rotation_only':'Rotation FP16','unrotated_4bit':'Unrotated W4A4KV4','quarot_4bit':'QuaRot W4A4KV4'}
metrics={n:json.loads((ROOT/'results'/n/'metrics.json').read_text()) for n in NAMES if (ROOT/'results'/n/'metrics.json').exists()}
summary={}
if metrics:
 fig,ax=plt.subplots(figsize=(9,4))
 ns=list(metrics); vals=[metrics[n]['wikitext2_ppl'] for n in ns]
 colors={'baseline':'#4972a4','rotation_only':'#8b9eb6','unrotated_4bit':'#d76b5f','quarot_4bit':'#24a478'}
 bars=ax.bar([LABELS[n] for n in ns],vals,color=[colors[n] for n in ns])
 ax.bar_label(bars,labels=[f'{v:.2e}' if v>1e5 else f'{v:.3f}' for v in vals]);ax.set_ylabel('WikiText-2 perplexity (lower is better)')
 ax.set_title('SmolLM2-1.7B · full test set · sequence length 2048')
 if max(vals)>10*min(vals):
  ax.set_yscale('log');ax.set_ylim(min(vals)*.6,max(vals)*2)
  ax.set_ylabel('WikiText-2 perplexity · log scale (lower is better)')
 else:
  ax.set_ylim(0,max(vals)*1.18)
 fig.tight_layout();fig.savefig(OUT/'perplexity.png',dpi=180);plt.close(fig)
 stable=[n for n in ['baseline','rotation_only','quarot_4bit'] if n in metrics]
 if len(stable)==3:
  fig,ax=plt.subplots(figsize=(8,4));v=[metrics[n]['wikitext2_ppl'] for n in stable]
  bars=ax.bar([LABELS[n] for n in stable],v,color=['#4972a4','#8b9eb6','#24a478'])
  ax.bar_label(bars,fmt='%.3f');ax.set_ylim(0,max(v)*1.2)
  ax.set_ylabel('WikiText-2 perplexity (lower is better)')
  ax.set_title('Quality gap among configurations with usable language performance')
  fig.tight_layout();fig.savefig(OUT/'perplexity_quality_gap.png',dpi=180);plt.close(fig)
 base=metrics.get('baseline',{}).get('wikitext2_ppl')
 for n,m in metrics.items():
  summary[n]={'ppl':m['wikitext2_ppl'],'seconds':m['elapsed_seconds'],
              'peak_gpu_allocated_gib':m['peak_cuda_allocated_bytes']/2**30}
  if base:
   summary[n].update(ppl_delta=m['wikitext2_ppl']-base,ppl_relative_change_percent=100*(m['wikitext2_ppl']/base-1))

def qdq(x):
 scale=(x.abs().amax(-1,keepdim=True)*.9/7).clamp_min(1e-8)
 return (x/scale).round().clamp(-8,7)*scale

def stats(x):
 x=x.float(); q=qdq(x); mse=(x-q).square().mean().item();power=x.square().mean().item()
 return {'abs_max':x.abs().max().item(),'rms':math.sqrt(power),'max_over_rms':x.abs().max().item()/math.sqrt(power),
  'abs_p99':torch.quantile(x.abs().flatten(),.99).item(),'int4_mse':mse,'int4_nmse':mse/power,'int4_sqnr_db':10*math.log10(power/mse)}

base_path=ROOT/'results'/'baseline'/'ffn_input.pt'
rot_path=ROOT/'results'/'rotation_only'/'ffn_input.pt'
if base_path.exists() and rot_path.exists():
 a=torch.load(base_path,map_location='cpu');b=torch.load(rot_path,map_location='cpu')
 scale_path=ROOT/'results'/'rotation_only'/'ffn_norm_scale.pt'
 plot_a=a/torch.load(scale_path,map_location='cpu') if scale_path.exists() else a
 fig,axs=plt.subplots(1,2,figsize=(13,4),sharey=True)
 for ax,x,label in zip(axs,[plot_a,b],['Before rotation (norm scale removed)','After QuaRot (rotation only)']):
  p=torch.quantile(x,torch.tensor([0.,.01,.25,.75,.99,1.]),dim=0).numpy();idx=np.arange(x.shape[1])
  ax.fill_between(idx,p[0],p[5],alpha=.2,label='Min–max');ax.fill_between(idx,p[1],p[4],alpha=.4,label='1st–99th percentile')
  ax.plot(idx,p[0],color='#4972a4',linewidth=.45,alpha=.75)
  ax.plot(idx,p[5],color='#4972a4',linewidth=.45,alpha=.75)
  ax.fill_between(idx,p[2],p[3],alpha=.8,label='25th–75th percentile');ax.set_title(label);ax.set_xlabel('Hidden channel')
 axs[0].set_ylabel('Activation');axs[1].legend(fontsize=8)
 fig.suptitle('Tenth block · same 256 WikiText-2 token positions · shared axes')
 fig.tight_layout();fig.savefig(OUT/'activation_envelopes.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,4),sharey=True)
 vmax=max(plot_a.abs().max().item(),b.abs().max().item())
 for ax,x,label in zip(axs,[plot_a,b],['Before rotation (norm scale removed)','Rotated']):
  im=ax.imshow(x[:64].numpy(),aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
  ax.set_title(label);ax.set_xlabel('Hidden channel')
 axs[0].set_ylabel('Token position');fig.colorbar(im,ax=axs.ravel().tolist(),label='Activation')
 fig.savefig(OUT/'activation_heatmap.png',dpi=180,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4))
 for x,label in [(plot_a,'Before rotation (norm scale removed)'),(b,'Rotated')]:
  ax.hist(x.abs().flatten().numpy(),bins=np.geomspace(1e-6,max(vmax,1e-5),100),histtype='step',density=True,label=label)
 ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Absolute activation');ax.set_ylabel('Density');ax.legend()
 fig.tight_layout();fig.savefig(OUT/'activation_distribution.png',dpi=180);plt.close(fig)
 summary['activation_hook_comparison']={'original':stats(a),'rotated':stats(b),
 'caveat':'Original post-norm scale is fused into weights for rotated model; hook values include parameterization change.'}
 qpath=ROOT/'results'/'rotation_only'/'rotation_matrix.pt'
 scale_path=ROOT/'results'/'rotation_only'/'ffn_norm_scale.pt'
 if qpath.exists() and scale_path.exists():
  scale=torch.load(scale_path,map_location='cpu')
  # Same layer norm weights in the unmodified model and at rotated-model load time.
  normed=a.double()/scale.double()
  q=torch.load(qpath,map_location='cpu').double()
  rotated=normed@q
  summary['isolated_rotation']={'original_normalized':stats(normed),'directly_rotated':stats(rotated),
    'relative_norm_error':((normed.norm()-rotated.norm()).abs()/normed.norm()).item(),
    'relative_activation_agreement_error':((rotated-b.double()).norm()/rotated.norm()).item()}
  fig,axs=plt.subplots(1,3,figsize=(12,3.5))
  for ax,key,title in zip(axs,['max_over_rms','int4_nmse','int4_sqnr_db'],
                         ['Max / RMS (lower is better)','INT4 normalized MSE (lower is better)','INT4 SQNR, dB (higher is better)']):
   s=summary['isolated_rotation'];values=[s['original_normalized'][key],s['directly_rotated'][key]]
   bars=ax.bar(['Before','Rotated'],values,color=['#4972a4','#24a478']);ax.bar_label(bars,fmt='%.3f')
   ax.set_title(title,fontsize=10);ax.set_ylim(0,max(values)*1.22)
  fig.suptitle('Same normalized activations · isolated rotation · 4-bit clipping ratio 0.9')
  fig.tight_layout();fig.savefig(OUT/'rotation_quantization_effect.png',dpi=180);plt.close(fig)
 base_logits=ROOT/'results'/'baseline'/'logits.pt';rot_logits=ROOT/'results'/'rotation_only'/'logits.pt'
 if base_logits.exists() and rot_logits.exists():
  la=torch.load(base_logits,map_location='cpu');lb=torch.load(rot_logits,map_location='cpu')
  summary['rotation_logit_agreement']={'max_absolute_error':(la-lb).abs().max().item(),
    'relative_l2_error':((la-lb).norm()/la.norm()).item(),
    'top1_agreement_percent':(la.argmax(-1)==lb.argmax(-1)).float().mean().item()*100,
    'sampled_token_positions':la.shape[0]}
tasks=['piqa','winogrande','hellaswag','arc_easy','arc_challenge','lambada_openai']
zs={}
for n in ['baseline','quarot_4bit']:
 path=ROOT/'results'/n/'zero_shot.json'
 if path.exists():
  full=json.loads(path.read_text());raw=full['results']
  zs[n]={t:{'metric':'acc_norm,none' if 'acc_norm,none' in raw[t] else 'acc,none',
            'accuracy_percent':100*raw[t].get('acc_norm,none',raw[t]['acc,none']),
            'examples':len(full.get('samples',{}).get(t,[]))} for t in tasks}
  zs[n]['mean_accuracy_percent']=sum(zs[n][t]['accuracy_percent'] for t in tasks)/len(tasks)
if len(zs)==2:
 zs['accuracy_retention_percent']=100*zs['quarot_4bit']['mean_accuracy_percent']/zs['baseline']['mean_accuracy_percent']
 zs['mean_accuracy_delta_pp']=zs['quarot_4bit']['mean_accuracy_percent']-zs['baseline']['mean_accuracy_percent']
 fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(tasks));w=.36
 for offset,n in [(-w/2,'baseline'),(w/2,'quarot_4bit')]:
  ax.bar(x+offset,[zs[n][t]['accuracy_percent'] for t in tasks],w,label=LABELS[n])
 ax.set_xticks(x,['PIQA','WinoGrande','HellaSwag','ARC-Easy','ARC-Challenge','LAMBADA'])
 ax.set_ylim(0,100);ax.set_ylabel('Zero-shot accuracy (%)');ax.legend();fig.tight_layout()
 fig.savefig(OUT/'zero_shot_accuracy.png',dpi=180);plt.close(fig)
if zs: summary['zero_shot']=zs
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
report=ROOT/'REPORT.md'
text=report.read_text().split('\n<!-- MEASURED_RESULTS -->')[0]
status='All four perplexity configurations and both full six-task zero-shot evaluations completed.' if len(metrics)==4 and 'baseline' in zs and 'quarot_4bit' in zs else 'Accuracy reproduction in progress; the measured table below contains completed runs only.'
text=__import__('re').sub(r'Status: [^\n]+', 'Status: '+status, text)
text=text.replace('Full task evaluation remains pending; do not replace it with generated examples or partial task scores without clear labels.', 'Use full evaluation splits; the completion state and measured scores appear below.')
text=text.replace('Results and figures will be added after successful runs. Record failures and runtimes without representing them as completed reproduction evidence.', 'The measured results and figures below are generated from completed runs. One calibration/rotation seed and one layer’s 256-token activation sample limit the scope of the conclusions.')
text+='\n<!-- MEASURED_RESULTS -->\n## Measured results\n\n'
text+='| Configuration | WikiText-2 PPL ↓ | Change from FP16 | Elapsed (s) | Peak allocated VRAM (GiB) |\n|---|---:|---:|---:|---:|\n'
for n in NAMES:
 if n in metrics:
  m=summary[n];text+=f"| {LABELS[n]} | {m['ppl']:.4f} | {m.get('ppl_delta',0):+.4f} | {m['seconds']:.1f} | {m['peak_gpu_allocated_gib']:.2f} |\n"
text+='\nTimings describe floating-point accuracy simulation, including setup, and are not INT4 inference speedups. GPU figures are peak PyTorch allocations, excluding display/other-process memory.\n\n'
text+='The QuaRot perplexity run used the original per-call allocator cleanup; the control used the optimized cleanup placement. These elapsed times therefore also reflect implementation and scheduling differences and must not be compared as algorithm speed benchmarks.\n\n'
if 'unrotated_4bit' in summary and 'quarot_4bit' in summary:
 u=summary['unrotated_4bit'];q=summary['quarot_4bit']
 text+=f"The unrotated W4A4KV4 control collapsed (PPL {u['ppl']:,.0f}). QuaRot prevented this collapse, but its PPL remains {q['ppl_relative_change_percent']:.1f}% above FP16. Perplexity can exceed vocabulary size when the model assigns very small probabilities to correct tokens; this result is finite, not numerical overflow.\n\n"
if 'rotation_logit_agreement' in summary:
 m=summary['rotation_logit_agreement'];text+=f"Rotation-only check on {m['sampled_token_positions']} token positions: relative logit L2 error {m['relative_l2_error']:.6f}, maximum absolute error {m['max_absolute_error']:.4f}, top-1 agreement {m['top1_agreement_percent']:.1f}%.\n\n"
if 'isolated_rotation' in summary:
 m=summary['isolated_rotation'];a=m['original_normalized'];b=m['directly_rotated']
 text+=f"On the same normalized activation tensor, rotation changes max/RMS from {a['max_over_rms']:.2f} to {b['max_over_rms']:.2f}, normalized INT4 reconstruction MSE from {a['int4_nmse']:.6f} to {b['int4_nmse']:.6f}, and SQNR from {a['int4_sqnr_db']:.2f} to {b['int4_sqnr_db']:.2f} dB. This controlled local reconstruction test isolates rotation from RMSNorm scale fusion; it is distinct from end-to-end quantized model accuracy.\n\n"
if len(zs)>=2 and 'baseline' in zs and 'quarot_4bit' in zs:
 text+='| Task | Examples | Metric | FP16 (%) | QuaRot (%) | Difference (pp) |\n|---|---:|---|---:|---:|---:|\n'
 for t in tasks:
  a=zs['baseline'][t];b=zs['quarot_4bit'][t];text+=f"| {t} | {a['examples']} | {a['metric']} | {a['accuracy_percent']:.2f} | {b['accuracy_percent']:.2f} | {b['accuracy_percent']-a['accuracy_percent']:+.2f} |\n"
 text+=f"\nMean accuracy: FP16 {zs['baseline']['mean_accuracy_percent']:.2f}%, QuaRot {zs['quarot_4bit']['mean_accuracy_percent']:.2f}%; retention {zs['accuracy_retention_percent']:.2f}%.\n\n"
else:
 text+='Full six-task zero-shot evaluation is pending. No accuracy numbers are inferred from perplexity.\n\n'
for filename,label in [('perplexity.png','Perplexity'),('perplexity_quality_gap.png','FP16 versus QuaRot quality gap'),('activation_envelopes.png','Channel distributions'),('activation_heatmap.png','Activation heatmap'),('activation_distribution.png','Magnitude distribution'),('rotation_quantization_effect.png','Isolated rotation and quantization error'),('zero_shot_accuracy.png','Zero-shot accuracy')]:
 if (OUT/filename).exists(): text+=f'![{label}](figures/{filename})\n\n'
report.write_text(text)
print(json.dumps(summary,indent=2))
