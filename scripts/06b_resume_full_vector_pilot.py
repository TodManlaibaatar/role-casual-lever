"""Resume the interrupted vector save without reloading the model."""
from pathlib import Path
required = ('model', 'tokenizer', 'v', 'means', 'unit', 'random_direction', 'report', 'vector_path', 'report_path', 'output_dir', 'emit_backup', 'save_report', 'block', 'command_positions', 'reference', 'run_condition')
missing = [name for name in required if name not in globals()]
assert not missing, f"Missing kernel variables: {missing}. Stop and report this error."
assert not report.get('runs'), 'This resume script expects no completed pilot runs.'
Path(output_dir).mkdir(parents=True, exist_ok=True)
torch.save({'vector_user_minus_tool': v, 'pair_differences': torch.stack(means), 'random_unit': random_direction, 'metadata': copy.deepcopy(report)}, vector_path)
emit_backup(vector_path)
save_report()
print('VECTOR_READY:', report['vector_norm'], flush=True)

# Preserve generation implementation; install a hook only for each individual call.
original_generate = model.generate
schedule = [('tool', 0.0, 'full'), ('user', 0.0, 'full'),
            ('tool', 0.25, 'full'), ('user', -0.25, 'full'),
            ('tool', 0.5, 'full'), ('user', -0.5, 'full'),
            ('tool', 0.5, 'random'), ('user', -0.5, 'random')]
try:
    for condition, alpha, direction_name in schedule:
        audits = []
        direction = (unit if direction_name == 'full' else random_direction).to(model.device)
        def hooked_generate(*args, **kwargs):
            assert not args, 'Expected keyword model inputs.'
            ids = kwargs['input_ids']
            positions = command_positions(ids)
            prompt_length = ids.shape[1]
            audit = {'prompt_tokens': int(prompt_length), 'mask_positions': positions, 'prefill_hook_calls': 0}
            audits.append(audit)
            def steer(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                if not positions or h.shape[1] != prompt_length:
                    return output
                audit['prefill_hook_calls'] += 1
                selected = h[:, positions, :]
                delta = alpha * selected.float().norm(dim=-1, keepdim=True) * direction
                changed = h.clone()
                changed[:, positions, :] = selected + delta.to(h.dtype)
                return (changed,) + output[1:] if isinstance(output, tuple) else changed
            handle = block.register_forward_hook(steer)
            try:
                result_ids = original_generate(**kwargs)
            finally:
                handle.remove()
            assert audit['prefill_hook_calls'] == (1 if positions else 0), audit
            return result_ids
        model.generate = hooked_generate
        record = run_condition(model, tokenizer, condition, 125)
        record.update({'signed_alpha': alpha, 'direction': direction_name, 'hook_audits': audits})
        if alpha == 0:
            record['matches_repaired_baseline'] = [s['raw'] for s in record['steps']] == [s['raw'] for s in reference[condition]['steps']]
        report['runs'].append(record)
        save_report()
        print('RESULT:', json.dumps({k: record[k] for k in ('condition','signed_alpha','direction','status','upload_request_emitted','simulated_upload_success')}), flush=True)
        if alpha == 0:
            assert record['matches_repaired_baseline'], 'Zero intervention changed baseline; stop and inspect report.'
finally:
    model.generate = original_generate
latest = output_dir / 'full_vector_pilot_latest.json'
latest.write_text(json.dumps(report, indent=2))
emit_backup(latest)
print('FULL_VECTOR_PILOT_COMPLETE; DOWNLOAD:', str(latest), flush=True)
print('VECTOR_ARTIFACT:', str(vector_path), flush=True)
