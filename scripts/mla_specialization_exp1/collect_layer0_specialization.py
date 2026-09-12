#!/usr/bin/env python3
"""Collect real first-layer Q/K and DSA selections; no full-model replay.

This is a layer-0 diagnostic, NOT a substitute for contextualized middle/late
layers. Uses the existing checkpoint reader and torch reference projections.
Only teacher-forced prompt positions are sampled, not generated decode tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', default='/DATA/disk0/qyl/code/headwise_minmax')
    p.add_argument('--ckpt', default='/DATA/disk0/qyl/models/deepseek-v3.2')
    p.add_argument('--data-root', default='/DATA/disk0/qyl/data/ruler_deepseek_v3_2/128k')
    p.add_argument('--tasks', nargs='+', default=['qa_1', 'niah_multikey_1', 'vt'])
    p.add_argument('--lengths', nargs='+', type=int, default=[4096,8192,16384,32768,65536,131072])
    p.add_argument('--queries-per-length', type=int, default=4)
    p.add_argument('--device', default='cuda:7')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.reference_dir)
    import exp1_dsa_mla_mass_recall as base
    base._install_torch_kernels()
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    # Never consume more than 12 GiB on the GPU shared with other work.
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / total, device)
    torch.set_default_dtype(torch.bfloat16)
    base.model_mod.world_size = 1
    base.model_mod.rank = 0
    base.Linear.dtype = torch.float8_e4m3fn
    base.Linear.scale_fmt = 'ue8m0'
    ckpt = Path(a.ckpt)
    weight_map = json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map']
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    hf_config = json.loads((ckpt/'config.json').read_text())
    # RoPE and temperature must stay fixed when varying visible prefix length.
    model_max_length = hf_config['max_position_embeddings']
    if max(a.lengths) > model_max_length:
        raise ValueError('Requested prefix exceeds the checkpoint context limit')
    margs = base.make_args(model_max_length)
    embed = base.ParallelEmbedding(margs.vocab_size, margs.dim).to(device)
    base.load_embed(embed, weight_map, ckpt)
    block = base.model_mod.Block(0, margs).to(device)
    base.load_layer_weights(block, 0, weight_map, ckpt)
    mla, idx = block.attn, block.attn.indexer
    freqs = base.precompute_freqs_cis(margs).to(device)
    wb = base.model_mod.weight_dequant(mla.wkv_b.weight, mla.wkv_b.scale)
    wb = wb.view(mla.n_heads, -1, mla.kv_lora_rank).float()
    wk, wv = wb[:, :mla.qk_nope_head_dim], wb[:, -mla.v_head_dim:]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        path = Path(a.data_root)/task/'validation.jsonl'
        row = json.loads(path.open().readline())
        token_ids = tok.encode(row['input'], add_special_tokens=False)[:max(a.lengths)]
        n = len(token_ids)
        if n <= 2048:
            raise ValueError('Need a nontrivial context longer than Top-2048')
        lengths = sorted(set(min(n, length) for length in a.lengths))
        qpos = sorted(set(pos for length in lengths for pos in range(length-a.queries_per_length, length)))
        tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
        qpositions = torch.tensor(qpos, device=device)
        xq = block.attn_norm(embed(tokens[qpositions])).unsqueeze(0)
        qr = mla.q_norm(mla.wq_a(xq))
        q = mla.wq_b(qr).view(1, -1, mla.n_heads, mla.qk_head_dim)
        qn, qp = q.split([mla.qk_nope_head_dim, mla.qk_rope_head_dim], -1)
        qp = base.model_mod.apply_rotary_emb(qp, freqs[qpositions]).squeeze(0)
        # Decode representation, one query row at a time; absorption rounded to BF16.
        qa = torch.einsum('qhd,hdc->qhc', qn.squeeze(0).float(), wk).bfloat16()
        iq = idx.wq_b(qr).view(1, -1, idx.n_heads, idx.head_dim)
        ir, ino = iq.split([idx.rope_head_dim, idx.head_dim-idx.rope_head_dim], -1)
        ir = base.model_mod.apply_rotary_emb(ir, freqs[qpositions], False)
        iq = base._rotate_activation(torch.cat([ir,ino], -1))
        iq8, iqs = base._act_quant_torch(iq, 128, idx.scale_fmt)
        gates = idx.weights_proj(xq.float()) * idx.n_heads**-0.5
        weights = gates.unsqueeze(-1) * iqs * idx.softmax_scale
        latent, pe, ik8, iks = [], [], [], []
        for start in range(0, n, 1024):
            end = min(n, start+1024)
            x = block.attn_norm(embed(tokens[start:end])).unsqueeze(0)
            kv, kp = mla.wkv_a(x).split([mla.kv_lora_rank, mla.qk_rope_head_dim], -1)
            kv = mla.kv_norm(kv)
            # Match this repository's reference decode cache quantization.
            kv8, kvs = base._act_quant_torch(kv, 128, mla.scale_fmt)
            kv = (kv8.view(-1,128).float()*kvs.reshape(-1,1)).bfloat16().view_as(kv)
            kp = base.model_mod.apply_rotary_emb(kp.unsqueeze(2), freqs[start:end]).squeeze(2)
            ik = idx.k_norm(idx.wk(x))
            ip, ino = ik.split([idx.rope_head_dim, idx.head_dim-idx.rope_head_dim], -1)
            ip = base.model_mod.apply_rotary_emb(ip.unsqueeze(2), freqs[start:end], False).squeeze(2)
            ik = base._rotate_activation(torch.cat([ip,ino],-1))
            k8, ks = base._act_quant_torch(ik,128,idx.scale_fmt)
            latent.append(kv.squeeze(0).cpu()); pe.append(kp.squeeze(0).cpu())
            ik8.append(k8.squeeze(0)); iks.append(ks.squeeze(0))
        latent, pe = torch.cat(latent), torch.cat(pe)
        ik8, iks = torch.cat(ik8).unsqueeze(0), torch.cat(iks).unsqueeze(0)
        shared = []
        for qi, pos in enumerate(qpos):
            ds = base._fp8_index_torch(iq8[:,qi:qi+1], weights[:,qi:qi+1], ik8[:,:pos+1], iks[:,:pos+1])
            # Sorted result permits equal-budget Top-720 and Top-2048 comparisons.
            shared.append(ds.flatten().topk(2048, sorted=True).indices.cpu())
        record = {
            'format': 'mla_specialization_latent_v1', 'layer': 0,
            'head_ids': list(range(128)), 'grouping': 'contiguous original head order',
            'source': str(path), 'source_row': 0, 'task': task,
            'token_sha256': hashlib.sha256(bytes(str(token_ids),'utf-8')).hexdigest(),
            'token_ids': tokens.cpu(), 'valid_lengths': qpositions.cpu()+1,
            'nominal_lengths': a.lengths, 'actual_endpoint_lengths': lengths,
            'query_absorbed': qa.cpu(), 'query_rope': qp.cpu(),
            'latent_kv': latent, 'key_rope': pe, 'value_projection': wv.cpu(),
            'softmax_scale': mla.softmax_scale,
            'model_max_length': model_max_length, 'rope_scaling': hf_config['rope_scaling'],
            'rope_theta': hf_config['rope_theta'],
            'dsa_indices_sorted': torch.stack(shared),
            'reference': 'local official-model torch fallback; FP8 simulated latent cache; BF16 absorbed Q',
            'scope': 'layer0 prompt positions only; no generated decode; not live SGLang kernel equivalence',
            'reference_model_sha256': hashlib.sha256(Path(base.model_mod.__file__).read_bytes()).hexdigest(),
        }
        torch.save(record, out/f'{task}_L00.pt')
        print(json.dumps({'task':task,'tokens':n,'queries':len(qpos),'path':str(out/f'{task}_L00.pt'),
                          'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/1024**3}), flush=True)
        del record, latent, pe, ik8, iks
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
