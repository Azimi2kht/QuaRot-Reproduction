"""Run upstream accuracy pipeline using local mirror and save measured artifacts."""
import json, os, sys, time, threading, resource, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BACKEND = os.environ.get('HADAMARD_BACKEND','reference')
if BACKEND not in ['reference','native']:
    raise ValueError('HADAMARD_BACKEND must be reference or native')
if BACKEND == 'reference':
    sys.path.insert(0, str(ROOT / 'scripts' / 'compat'))
sys.path.insert(0, str(ROOT / 'QuaRot' / 'fake_quant'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch, psutil, transformers
if os.environ.get('TASKS_ONLY') == '1':
    torch.cuda.set_per_process_memory_fraction(0.65)
import model_utils, data_utils, eval_utils, utils
utils.supported_models.append('HuggingFaceTB/SmolLM2-1.7B')

name = os.environ.get('EXPERIMENT_NAME', 'baseline')
outdir = ROOT / 'results' / name
outdir.mkdir(parents=True, exist_ok=True)
(outdir/'failure.json').unlink(missing_ok=True)
if name in ['quarot_4bit','unrotated_4bit'] and '--save_qmodel_path' not in sys.argv and '--load_qmodel_path' not in sys.argv:
    checkpoint = outdir/'model.pt'
    sys.argv.extend(['--load_qmodel_path' if checkpoint.exists() else '--save_qmodel_path', str(checkpoint)])
start = time.monotonic()
metrics = {'experiment':name, 'model':'SmolLM2-1.7B', 'source':'HuggingFaceTB/SmolLM2-1.7B',
 'revision':'effd688a12921b4cc83e3312b6feb579f70f9c71',
 'torch':torch.__version__, 'transformers':transformers.__version__,
 'hadamard_backend':('official fast-hadamard-transform 1.0.4.post1 CUDA wheel' if BACKEND == 'native' else 'PyTorch FWHT accuracy fallback; not a performance benchmark'),
 'peak_rss_bytes':0}
stop = threading.Event()
def monitor():
    while not stop.wait(1):
        metrics['peak_rss_bytes'] = max(metrics['peak_rss_bytes'], psutil.Process().memory_info().rss)
        (outdir/'progress.json').write_text(json.dumps({**metrics, 'elapsed_seconds':time.monotonic()-start}, indent=2))
threading.Thread(target=monitor, daemon=True).start()
local = str(ROOT/'models'/'SmolLM2-1.7B')
original_model = model_utils.get_model
original_loaders = data_utils.get_loaders

def load_model(model_name, hf_token=None):
    model = transformers.LlamaForCausalLM.from_pretrained(
        local, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation='sdpa' if os.environ.get('TASKS_ONLY') == '1' else 'eager')
    model.seqlen = 2048
    model.config.use_cache = False
    # Rotation transforms embedding and head separately; preserve tied weights by
    # cloning the head first, preventing the same tensor being rotated twice.
    if model.config.tie_word_embeddings:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False
    torch.save(model.model.layers[9].post_attention_layernorm.weight.detach().float().cpu(),
               outdir/'ffn_norm_scale.pt')
    print('Loaded local SmolLM2-1.7B, parameters:',sum(p.numel() for p in model.parameters()), flush=True)
    return model

def loaders(*args, **kwargs):
    kwargs['model'] = local
    result = original_loaders(*args, **kwargs)
    data_dir = ROOT/'data'
    data_dir.mkdir(exist_ok=True)
    if kwargs.get('eval_mode'):
        tensor = result.input_ids
        metrics['evaluation_token_sha256'] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        torch.save(tensor, data_dir/'wikitext2_test_tokens.pt')
    else:
        tensor = torch.cat([sample[0] for sample in result],dim=0)
        metrics['calibration_token_sha256'] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        torch.save(tensor, data_dir/'wikitext2_calibration_tokens.pt')
    return result
model_utils.get_model = load_model
data_utils.get_loaders = loaders
original_eval = eval_utils.evaluator

def evaluate(model, testenc, dev, args):
    if os.environ.get('TASKS_ONLY') == '1':
        return float('nan')
    # Training hooks must not be mistaken for evaluation activations.
    (outdir/'ffn_input.pt').unlink(missing_ok=True)
    def capture(module, inp):
        if not (outdir/'ffn_input.pt').exists():
            torch.save(inp[0][0,:256].detach().float().cpu(), outdir/'ffn_input.pt')
    model.model.layers[9].mlp.up_proj.register_forward_pre_hook(capture)
    def capture_logits(module, inp, output):
        if not (outdir/'logits.pt').exists():
            torch.save(output[0,:32].detach().float().cpu(), outdir/'logits.pt')
    model.lm_head.register_forward_hook(capture_logits)
    metrics['evaluation_sequences'] = testenc.input_ids.numel()//model.seqlen
    metrics['sequence_length'] = model.seqlen
    metrics['arguments'] = {k:v for k,v in vars(args).items() if k != 'hf_token'}
    t = time.monotonic()
    ppl = original_eval(model, testenc, dev, args)
    metrics.update(wikitext2_ppl=ppl, evaluation_seconds=time.monotonic()-t,
                   elapsed_seconds=time.monotonic()-start,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    metrics['peak_rss_bytes'] = max(metrics['peak_rss_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (outdir/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print('MEASURED_RESULT',json.dumps(metrics),flush=True)
    return ppl
eval_utils.evaluator = evaluate
try:
    if '--lm_eval' in sys.argv:
        import lm_eval
        original_simple_evaluate = lm_eval.simple_evaluate
        def save_task_results(*args, **kwargs):
            t = time.monotonic()
            kwargs['num_fewshot'] = 0
            if os.environ.get('TASK_LIMIT'):
                kwargs['limit'] = int(os.environ['TASK_LIMIT'])
            result = original_simple_evaluate(*args, **kwargs)
            (outdir/'zero_shot.json').write_text(json.dumps(result, indent=2, default=str))
            (outdir/'zero_shot_timing.json').write_text(json.dumps({
                'seconds':time.monotonic()-t, 'attention_backend':'sdpa',
                'batch_size':kwargs.get('batch_size'), 'num_fewshot':0,
                'example_limit_per_task':kwargs.get('limit'),
                'hadamard_backend':metrics['hadamard_backend'],
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'model_source':metrics['source'], 'model_revision':metrics['revision']}, indent=2))
            return result
        lm_eval.simple_evaluate = save_task_results
    if '--rotate' in sys.argv:
        import rotation_utils
        original_rotation_matrix = rotation_utils.get_orthogonal_matrix
        def save_rotation(*args, **kwargs):
            q = original_rotation_matrix(*args, **kwargs)
            torch.save(q.detach().cpu(), outdir/'rotation_matrix.pt')
            return q
        rotation_utils.get_orthogonal_matrix = save_rotation
    if name == 'unrotated_4bit':
        if '--rotate' in sys.argv:
            raise ValueError('Strict unrotated control cannot use --rotate')
        import rotation_utils
        # Upstream key wrapper otherwise applies a Q/K Hadamard even without --rotate.
        rotation_utils.hadamard_transform = lambda x, scale=1.0: x.contiguous()
    import main
    main.main()
except Exception as exc:
    (outdir/'failure.json').write_text(json.dumps({'error':type(exc).__name__, 'message':str(exc),
      'elapsed_seconds':time.monotonic()-start, 'peak_rss_bytes':metrics['peak_rss_bytes']},indent=2))
    raise
finally:
    stop.set()
