#!/usr/bin/env python3
"""Stream query rows from latent MLA captures into Experiment 1 metrics/plots."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from mla_specialization_metrics import compute_specialization_metrics


POLICIES = {'shared_dsa':'DSA shared', 'allH':'MLA shared oracle',
            'group16':'Fixed 16-head oracle', 'group2':'Pair oracle'}


def make_plots(groups, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for k in (720,2048):
        items = sorted((key,vals) for key,vals in groups.items() if key[2] == k)
        fig, axes = plt.subplots(2,3, figsize=(17,8), constrained_layout=True)
        for ax,(key,vals) in zip(axes.flat,items):
            layer, endpoint, _ = key
            for policy,label in POLICIES.items():
                masses = np.stack([v[f'{policy}_head_mass'] for v in vals]).mean(0).reshape(64,2).mean(1)
                ax.plot(np.arange(64),masses,label=label,linewidth=1.1)
            ax.set(title=f'L{layer} / {endpoint//1024}K bin / n={len(vals)}',
                   xlabel='MLA pair ID',ylabel='Captured dense attention mass',ylim=(0,1.03))
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'Experiment 1: equal K={k}; layer-0 prompt-prefix diagnostic')
        fig.savefig(out/f'pair_mass_K{k}.png',dpi=150)
        plt.close(fig)
        fig,axes = plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,(key,vals) in zip(axes.flat,items):
            matrix = np.stack([v['pair_overlap'] for v in vals]).mean(0)
            im = ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
            ax.set(title=f'L{key[0]} / {key[1]//1024}K prefix bin',xlabel='MLA pair ID',ylabel='MLA pair ID')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Oracle set intersection / K',shrink=.8)
        fig.suptitle(f'Pair-vs-pair probability-mass oracle overlap, K={k}')
        fig.savefig(out/f'pair_overlap_K{k}.png',dpi=150)
        plt.close(fig)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:7')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True,exist_ok=True)
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)
        torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(dev).total_memory,dev)
        torch.backends.cuda.matmul.allow_tf32 = False
    rows, groups = [], defaultdict(list)
    paths = sorted(Path(a.captures).glob('*.pt'))
    if not paths:
        raise ValueError('No capture files')
    for path in paths:
        c = torch.load(path,map_location='cpu',weights_only=True)
        if c['format'] != 'mla_specialization_latent_v1' or c['head_ids'] != list(range(128)):
            raise ValueError('Capture format or original head ordering mismatch')
        kv = c['latent_kv'].to(device=dev,dtype=torch.float32)
        pe = c['key_rope'].to(device=dev,dtype=torch.float32)
        qa = c['query_absorbed'].to(device=dev,dtype=torch.float32)
        qp = c['query_rope'].to(device=dev,dtype=torch.float32)
        wv = c['value_projection'].to(device=dev,dtype=torch.float32)
        for qi,length in enumerate(c['valid_lengths'].tolist()):
            logits = (qa[qi] @ kv[:length].T + qp[qi] @ pe[:length].T)*c['softmax_scale']
            probs = logits.softmax(-1)
            dsa = c['dsa_indices_sorted'][qi:qi+1].numpy()
            report, arrays = compute_specialization_metrics(probs.cpu().numpy()[None],np.array([length]),
                ks=(720,2048),shared_indices_by_k={720:dsa[:,:720],2048:dsa})
            dense = torch.einsum('hc,hdc->hd',probs @ kv[:length],wv)
            dense_norm = dense.norm()
            endpoint = next(x for x in c['actual_endpoint_lengths'] if x >= length)
            bucket = next(x for x in sorted(c['nominal_lengths']) if x >= endpoint)
            for k in (720,2048):
                rec = {'task':c['task'],'layer':c['layer'],'visible_length':length,'endpoint_length':endpoint,
                       'prefix_bucket':bucket,'k':k,'query_index':qi,'source_capture':str(path),'mass':{},'output_relative_l2':{}}
                agg = {}
                for policy in POLICIES:
                    mass = arrays[f'k{k}__{policy}__head_mass'][0]
                    agg[f'{policy}_head_mass'] = mass
                    rec['mass'][policy] = float(mass.mean())
                    inds = arrays[f'k{k}__{policy}__indices'][0]
                    hs = 128//len(inds)
                    sparse = torch.empty_like(dense)
                    for group,selected in enumerate(inds):
                        ts = torch.tensor(selected[selected>=0],device=dev)
                        sl = slice(group*hs,(group+1)*hs)
                        alpha = probs[sl][:,ts]
                        total = alpha.sum(-1,keepdim=True)
                        if (total<=0).any():
                            raise ValueError('Zero-mass selection: sparse output undefined')
                        latent_out = (alpha/total) @ kv[ts]
                        sparse[sl] = torch.einsum('hc,hdc->hd',latent_out,wv[sl])
                    # Before wo: pooled 128 head outputs, not full layer/residual error.
                    rec['output_relative_l2'][policy] = float((sparse-dense).norm()/dense_norm.clamp_min(1e-12))
                matrix = arrays[f'k{k}__pair_overlap'][0]
                agg['pair_overlap'] = matrix
                rec['pair_overlap_offdiag'] = float(matrix[~np.eye(64,dtype=bool)].mean())
                rec['pair_gain_vs_shared_mla'] = rec['mass']['group2']-rec['mass']['allH']
                rec['group16_gain_vs_shared_mla'] = rec['mass']['group16']-rec['mass']['allH']
                rec['query_pair_mass'] = {pol:agg[f'{pol}_head_mass'].reshape(64,2).mean(-1).tolist() for pol in POLICIES}
                rows.append(rec); groups[(c['layer'],bucket,k)].append(agg)
            # Compact exact index sets make every overlap/mass result inspectable.
            np.savez_compressed(out/f"{c['task']}_L{c['layer']:02}_q{qi:03}.npz",**arrays)
            print(json.dumps({key:rows[-1][key] for key in ('task','visible_length','mass','pair_gain_vs_shared_mla')}),flush=True)
        del kv,pe,qa,qp,wv,c
        if dev.type=='cuda': torch.cuda.empty_cache()
    summaries=[]
    for key,vals in sorted(groups.items()):
        layer,endpoint,k=key
        selected=[r for r in rows if (r['layer'],r['prefix_bucket'],r['k'])==key]
        summaries.append({'layer':layer,'prefix_bucket':endpoint,'k':k,'queries':len(selected),
            'actual_visible_length_range':[min(r['visible_length'] for r in selected),max(r['visible_length'] for r in selected)],
            'independent_prompts':len(set(r['task'] for r in selected)),
            'mean_mass':{pol:float(np.mean([r['mass'][pol] for r in selected])) for pol in POLICIES},
            'mean_output_relative_l2':{pol:float(np.mean([r['output_relative_l2'][pol] for r in selected])) for pol in POLICIES},
            'mean_pair_overlap_offdiag':float(np.mean([r['pair_overlap_offdiag'] for r in selected]))})
    result={'experiment':'Q1 only; no candidate assignment/budget sweep/training',
        'scope':'Layer0, 3 real RULER prompt prefixes; reference decode algebra, FP8 simulated KV. No middle/late layers or generated decode.',
        'oracle_objective':'TopK(mean_head softmax(logits)); maximize group-average dense attention mass',
        'dsa_baseline':'Recomputed reference weighted Indexer scores; sorted Top-720/2048 at identical budget',
        'output_metric':'Mean across queries of pooled per-head relative L2, after Wv but before wo; mass oracle is not output oracle',
        'aggregation':'Queries from each prompt are correlated; these are descriptive means, not independent-sample confidence intervals',
        'q1_gate':'UNRESOLVED: representative middle/late layers and generated decode prefixes still missing',
        'summaries':summaries,'queries':rows}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    make_plots(groups,out)


if __name__=='__main__': main()
