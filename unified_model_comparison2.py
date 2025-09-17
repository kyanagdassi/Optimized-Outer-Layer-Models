#!/usr/bin/env python3
"""
Unified Model Comparison 2 - Simplified version with proper float64 handling
"""

import os
import sys
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import gc

# Add the TFs_do_KF_ICL directory to the path
sys.path.append('TFs_do_KF_ICL/src')

from TFs_do_KF_ICL.src.models.gpt2 import GPT2
from TFs_do_KF_ICL.src.core.config import Config

def load_gpt2_model(checkpoint_path, n_dims_in=57, n_positions=251, n_embd=128, n_layer=12, n_head=8, n_dims_out=5):
    """Load GPT2 model from checkpoint and convert to float64"""
    print(f"Loading GPT2 model from checkpoint: {checkpoint_path}")
    
    # Load the model using the proper method
    gpt2_model = GPT2.load_from_checkpoint(
        checkpoint_path,
        n_dims_in=n_dims_in,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_dims_out=n_dims_out,
        map_location='cpu'
    )
    
    # Freeze backbone parameters
    for param in gpt2_model._backbone.parameters():
        param.requires_grad = False
    
    # Convert to float64
    print("Converting GPT2 model to double precision...")
    gpt2_model = gpt2_model.double()
    
    gpt2_model.eval()
    return gpt2_model

def compute_gpt2_activations(model, inputs, device='cpu'):
    """Compute GPT2 activations for inputs, preserving config structure"""
    model.eval()
    with torch.no_grad():
        # inputs shape: (n_configs, n_traces, T, D)
        n_configs, n_traces, T, D = inputs.shape
        activations = []
        
        for config_i in range(n_configs):
            config_activations = []
            for trace_i in range(n_traces):
                x = torch.tensor(inputs[config_i, trace_i:trace_i+1], dtype=torch.float64).to(device)
                embeds = model._read_in(x)
                hidden = model._backbone(inputs_embeds=embeds).last_hidden_state
                config_activations.append(hidden.cpu().numpy().astype(np.float64))
            
            # Concatenate traces for this config
            config_activations = np.concatenate(config_activations, axis=0)
            activations.append(config_activations)
        
        # Stack to get shape (n_configs, n_traces, T, n_embd)
        activations = np.stack(activations, axis=0).astype(np.float64)
        return activations

def compute_event_errors(predictions, targets, inputs, mask, config, n_dims_in, val_data):
    """Compute event errors using the same logic as unified_model_comparison.py"""
    # Event types: 5 after initial + 5 after final = 10 total
    event_types = [f"{k}_after_initial" for k in [1,2,3,7,8]] + [f"{k}_after_final" for k in [1,2,3,7,8]]
    
    n_events = len(event_types)
    n_configs, n_traces, T = predictions.shape[:3]  # predictions shape: (n_configs, n_traces, T, 5)
    event_errors = np.full((n_configs, n_traces, n_events), np.nan)
    
    for config_i in range(n_configs):
        for trace_i in range(n_traces):
            # After initial: k positions after the first open parenthesis
            for k_idx, k in enumerate([1, 2, 3, 7, 8]):
                idx = k
                if idx < T:
                    pred = predictions[config_i, trace_i, idx]  # (5,)
                    target = targets[config_i, trace_i, idx]    # (5,)
                    sqerr = np.sum((pred - target) ** 2)
                    event_errors[config_i, trace_i, k_idx] = sqerr
            
            # After final: k positions after the last open parenthesis
            for k_idx, k in enumerate([1, 2, 3, 7, 8]):
                idx = T - 1 - k
                if idx >= 0:
                    pred = predictions[config_i, trace_i, idx]  # (5,)
                    target = targets[config_i, trace_i, idx]    # (5,)
                    sqerr = np.sum((pred - target) ** 2)
                    event_errors[config_i, trace_i, k_idx + 5] = sqerr
    
    # Medians over traces, then configs
    median_over_traces = np.nanmedian(event_errors, axis=1)  # (n_configs, n_events)
    final_median_events = np.nanmedian(median_over_traces, axis=0)  # (n_events,)
    
    return final_median_events

def main():
    print("🚀 Starting Unified Model Comparison 2")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load GPT2 model
    checkpoint_path = 'TFs_do_KF_ICL/src/DataRandomTransformer/step%3D99000.ckpt'
    gpt2_model = load_gpt2_model(checkpoint_path)
    gpt2_model = gpt2_model.to(device)
    
    # Initialize empty matrices (float64)
    print("🔍 Initializing empty matrices...")
    YA_T = np.zeros((5, 128), dtype=np.float64)  # (n_dims_out, n_embd)
    AA_T = np.zeros((128, 128), dtype=np.float64)  # (n_embd, n_embd)
    
    print(f"🔍 DEBUG: Initial YA_T dtype: {YA_T.dtype}, shape: {YA_T.shape}")
    print(f"🔍 DEBUG: Initial AA_T dtype: {AA_T.dtype}, shape: {AA_T.shape}")
    
    # Set up plotting
    print("📊 Setting up learning curve plots...")
    haystack_lengths = [1, 2, 5]
    event_types = [f"{k}_after_initial" for k in [1,2,3,7,8]] + [f"{k}_after_final" for k in [1,2,3,7,8]]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'red', 'blue', 'green', 'orange', 'purple']
    
    # Create 3 main plots (one for each haystack)
    main_plots = {}
    for hay_len in haystack_lengths:
        fig = plt.figure(figsize=(15, 8))
        plt.xlabel('Number of Training Traces')
        plt.ylabel('Event Error (MSE) - Log Scale')
        plt.title(f'Event Error Learning Curves: Haystack Length {hay_len}')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Initialize data storage for this haystack
        main_plots[hay_len] = {
            'fig': fig,
            'data': {
                'traces': [],
                'gpt2_modified': {event: [] for event in event_types}
            }
        }
    
    # Initialize event errors dictionary
    event_errors_dict = {}
    
    # Set up paths
    activations_cache_dir = 'TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k/activations_cache'
    os.makedirs(activations_cache_dir, exist_ok=True)
    
    # Get batch files
    batch_dir = 'TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k'
    batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith('batch_') and f.endswith('.pkl')])
    
    print(f"📋 Found {len(batch_files)} batch files")
    for i, batch_file in enumerate(batch_files[:5]):  # Show first 5
        print(f"   {i+1}: {batch_file}")
    if len(batch_files) > 5:
        print(f"   ... ({len(batch_files)-5} more batches)")
    
    # Process each batch
    total_traces_processed = 0
    
    for batch_idx, batch_file in enumerate(batch_files):
        print(f"\n{'='*50}")
        print(f"BATCH {batch_idx + 1}/{len(batch_files)}: Processing {batch_file}")
        print(f"{'='*50}")
        
        # Load batch data
        batch_path = os.path.join(batch_dir, batch_file)
        with open(batch_path, 'rb') as f:
            batch_data = pickle.load(f)
        
        if 'multi_sys_ys' in batch_data:
            batch_inputs = batch_data['multi_sys_ys']  # Shape: (1, 20000, 1, 251, 57)
            batch_inputs = batch_inputs[0, :, 0, :, :]  # Reshape to (20000, 251, 57)
            print(f" Successfully loaded batch with shape: {batch_inputs.shape}")
        else:
            print(f"❌ No 'multi_sys_ys' found in {batch_file}")
            continue
        
        # Extract inputs and create targets (same logic as unified_model_comparison.py)
        n_dims_out = 5  # Number of output dimensions
        n_dims_in = 57  # Number of input dimensions
        payload_flag_idx = n_dims_in - n_dims_out - 1  # 51
        
        # Get payloads and create targets
        payloads = batch_inputs[..., -n_dims_out:]  # Last 5 dimensions are payloads
        targets = np.zeros_like(payloads, dtype=np.float64)
        targets[:, :-1, :] = payloads[:, 1:, :].astype(np.float64)
        
        print(f"📊 Target creation:")
        print(f"    Payloads shape: {payloads.shape}")
        print(f"    Payloads range: [{payloads.min():.6f}, {payloads.max():.6f}]")
        print(f"    Targets shape: {targets.shape}")
        print(f"    Targets range: [{targets.min():.6f}, {targets.max():.6f}]")
        print(f"    Non-zero targets: {np.count_nonzero(targets)}")
        
        # Create mask
        # We want to KEEP positions where payload_flag != 0 (unmasked positions)
        mask = np.ones((batch_inputs.shape[0], batch_inputs.shape[1]), dtype=bool)
        mask[:, :-1] = (batch_inputs[:, 1:, payload_flag_idx] != 0)  # Keep unmasked positions
        mask[:, -1] = True  # Always keep the last position
        
        print(f"📊 Batch data: {batch_inputs.shape[0]:,} traces, {batch_inputs.shape[1]} time steps")
        print(f"📊 Mask creation:")
        print(f"    Payload flag index: {payload_flag_idx}")
        print(f"    Payload flag values range: [{batch_inputs[:, 1:, payload_flag_idx].min()}, {batch_inputs[:, 1:, payload_flag_idx].max()}]")
        print(f"    Mask shape: {mask.shape}")
        print(f"    Total mask positions: {mask.size:,}")
        print(f"    Valid positions: {np.sum(mask):,}")
        print(f"    Invalid positions: {np.sum(~mask):,}")
        print(f"    Valid percentage: {100 * np.sum(mask) / mask.size:.1f}%")
        
        # Check if activations are cached
        # Extract batch number from filename like "batch_01_20k_traces.pkl" -> "01"
        batch_number = batch_file.split('_')[1]  # Extract "01" from "batch_01_20k_traces.pkl"
        cache_file = os.path.join(activations_cache_dir, f"training_batch_{batch_number}.npz")
        print(f"    🔍 Looking for cache file: {os.path.basename(cache_file)}")
        
        if os.path.exists(cache_file):
            print(f"    📁 Loading cached activations from: {os.path.basename(cache_file)}")
            try:
                cached_data = np.load(cache_file)
                batch_hidden_acts = cached_data['activations'].astype(np.float64)
                print(f"     Successfully loaded cached activations with shape: {batch_hidden_acts.shape}")
            except Exception as e:
                print(f"     ❌ Error loading cached activations: {e}")
                print(f"     🔄 Computing activations from scratch...")
                # Reshape batch_inputs to (1, n_traces, T, D) for compute_gpt2_activations
                batch_inputs_reshaped = batch_inputs.reshape(1, batch_inputs.shape[0], batch_inputs.shape[1], batch_inputs.shape[2])
                batch_hidden_acts = compute_gpt2_activations(gpt2_model, batch_inputs_reshaped, device)
                # Reshape back to (n_traces, T, n_embd)
                batch_hidden_acts = batch_hidden_acts[0]  # Remove the config dimension
        else:
            print(f"    🔄 Computing activations from scratch...")
            # Reshape batch_inputs to (1, n_traces, T, D) for compute_gpt2_activations
            batch_inputs_reshaped = batch_inputs.reshape(1, batch_inputs.shape[0], batch_inputs.shape[1], batch_inputs.shape[2])
            batch_hidden_acts = compute_gpt2_activations(gpt2_model, batch_inputs_reshaped, device)
            # Reshape back to (n_traces, T, n_embd)
            batch_hidden_acts = batch_hidden_acts[0]  # Remove the config dimension
            
            # Cache the activations
            print(f"    💾 Caching activations...")
            try:
                np.savez_compressed(
                    cache_file,
                    activations=batch_hidden_acts.astype(np.float64),
                    timestamp=time.time()
                )
                print(f"     Activations cached successfully")
            except Exception as e:
                print(f"     Warning: Could not cache activations: {e}")
        
        # Update cumulative matrices (chunked for memory efficiency)
        print(f"🔄 Updating cumulative matrices...")
        
        # Reshape for matrix operations
        n_traces, T, n_embd = batch_hidden_acts.shape
        batch_hidden_acts_flat = batch_hidden_acts.reshape(-1, n_embd)  # (n_traces*T, n_embd)
        targets_flat = targets.reshape(-1, n_dims_out)  # (n_traces*T, n_dims_out)
        mask_flat = mask.reshape(-1)  # (n_traces*T,)
        
        # Apply mask
        valid_indices = mask_flat
        A_current_unmasked = batch_hidden_acts_flat[valid_indices]  # (N_valid, n_embd)
        Y_current_unmasked = targets_flat[valid_indices].T  # (n_dims_out, N_valid)
        
        print(f"    📊 Masking statistics:")
        print(f"        Total positions: {mask_flat.shape[0]:,}")
        print(f"        Valid positions: {np.sum(valid_indices):,}")
        print(f"        Invalid positions: {np.sum(~valid_indices):,}")
        print(f"        Valid percentage: {100 * np.sum(valid_indices) / mask_flat.shape[0]:.1f}%")
        print(f"    📊 A_batch shape: {A_current_unmasked.shape}")
        print(f"    📊 Y_batch shape: {Y_current_unmasked.shape}")
        print(f"    📊 Using {A_current_unmasked.shape[0]:,} valid positions for matrix update")
        
        # Chunked processing for memory efficiency (same as unified_model_comparison.py)
        chunk_size = 10000  # Process in chunks to avoid memory issues
        N_valid = A_current_unmasked.shape[0]
        num_chunks = (N_valid + chunk_size - 1) // chunk_size
        
        print(f"    🔄 Processing matrix products in {num_chunks} chunks of {chunk_size} columns...")
        
        # Initialize accumulation matrices
        matrix_product_YA = np.zeros((5, 128), dtype=np.float64)
        matrix_product_AA = np.zeros((128, 128), dtype=np.float64)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, N_valid)
            
            # Get chunk data
            A_chunk = A_current_unmasked[start_idx:end_idx]  # (chunk_size, 128)
            Y_chunk = Y_current_unmasked[:, start_idx:end_idx]  # (5, chunk_size)
            
            # Compute chunk products
            chunk_YA = Y_chunk @ A_chunk  # (5, 128)
            chunk_AA = A_chunk.T @ A_chunk  # (128, 128)
            
            # Debug: Print chunk statistics
            if chunk_idx == 0:  # Only print for first chunk to avoid spam
                print(f"        🔍 First chunk debug:")
                print(f"            A_chunk shape: {A_chunk.shape}")
                print(f"            Y_chunk shape: {Y_chunk.shape}")
                print(f"            chunk_YA shape: {chunk_YA.shape}, range: [{chunk_YA.min():.6f}, {chunk_YA.max():.6f}]")
                print(f"            chunk_AA shape: {chunk_AA.shape}, range: [{chunk_AA.min():.6f}, {chunk_AA.max():.6f}]")
                print(f"            Y_chunk range: [{Y_chunk.min():.6f}, {Y_chunk.max():.6f}]")
                print(f"            A_chunk range: [{A_chunk.min():.6f}, {A_chunk.max():.6f}]")
            
            # Accumulate
            matrix_product_YA += chunk_YA
            matrix_product_AA += chunk_AA
            
            # Memory cleanup
            del A_chunk, Y_chunk, chunk_YA, chunk_AA
            gc.collect()
        
        # Update cumulative matrices
        print(f"    🔍 Matrix products before update:")
        print(f"        matrix_product_YA range: [{matrix_product_YA.min():.6f}, {matrix_product_YA.max():.6f}]")
        print(f"        matrix_product_AA range: [{matrix_product_AA.min():.6f}, {matrix_product_AA.max():.6f}]")
        
        YA_T += matrix_product_YA.astype(np.float64)  # 5 x 128
        AA_T += matrix_product_AA.astype(np.float64)  # 128 x 128
        
        print(f"    📊 Updated matrices: YA_T {YA_T.shape}, AA_T {AA_T.shape}")
        print(f"    🔍 YA_T range: [{YA_T.min():.6f}, {YA_T.max():.6f}]")
        print(f"    🔍 AA_T range: [{AA_T.min():.6f}, {AA_T.max():.6f}]")
        
        # Compute W_opt
        print(f"🔍 Computing W_opt...")
        try:
            W_opt = YA_T @ np.linalg.inv(AA_T)  # (n_embd, n_dims_out)
            print(f"    ✅ W_opt computed successfully: {W_opt.shape}")
        except Exception as e:
            print(f"    ❌ Error computing W_opt: {e}")
            W_opt = np.zeros((5, 128), dtype=np.float64)
        
        # Test on haystacks
        print(f"🧪 Testing on haystacks...")
        haystack_lengths = [1, 2, 5]
        haystack_results = {}
        
        for hay_len in haystack_lengths:
            print(f"  Testing haystack length {hay_len}...")
            
            try:
                # Check if haystack activations are cached
                haystack_cache_file = os.path.join(activations_cache_dir, f"haystack_{hay_len}_validation.npz")
                
                if os.path.exists(haystack_cache_file):
                    print(f"    📁 Using cached haystack {hay_len} activations")
                    cached_data = np.load(haystack_cache_file)
                    val_inputs = cached_data['inputs']
                    val_targets = cached_data['targets']
                    val_mask = cached_data['masks']
                    gpt2_val_hidden_acts = cached_data['activations'].astype(np.float64)
                    
                else:
                    print(f"    🔄 Computing haystack {hay_len} activations from scratch...")
                    
                    # Load validation data
                    val_data_path = f'TFs_do_KF_ICL/src/DataRandomTransformer/val_interleaved_traces_ortho_haar_ident_C_haystack_len_{hay_len}.pkl'
                    with open(val_data_path, 'rb') as f:
                        val_data = pickle.load(f)
                    
                    val_inputs = val_data['multi_sys_ys']  # (n_configs, 1, n_traces, T, D)
                    n_configs, _, n_traces, T_val, D = val_inputs.shape
                    val_inputs = val_inputs.reshape(n_configs, n_traces, T_val, D)
                    
                    # Create targets and mask
                    val_payloads = val_inputs[..., -n_dims_out:]
                    val_targets = np.zeros_like(val_payloads, dtype=np.float64)
                    val_targets[:, :, :-1, :] = val_payloads[:, :, 1:, :].astype(np.float64)
                    
                    val_mask = np.ones((n_configs, n_traces, T_val), dtype=bool)
                    val_mask[:, :, :-1] = (val_inputs[:, :, 1:, payload_flag_idx] != 0)  # Keep unmasked positions
                    val_mask[:, :, -1] = True
                    
                    # Compute activations (preserving config structure)
                    gpt2_val_hidden_acts = compute_gpt2_activations(gpt2_model, val_inputs, device)
                    
                    # Cache the haystack data
                    print(f"    💾 Caching haystack {hay_len} activations...")
                    try:
                        np.savez_compressed(
                            haystack_cache_file,
                            inputs=val_inputs,
                            targets=val_targets,
                            masks=val_mask,
                            activations=gpt2_val_hidden_acts.astype(np.float64),
                            haystack_length=hay_len,
                            timestamp=time.time()
                        )
                        print(f"     Haystack {hay_len} activations cached")
                    except Exception as e:
                        print(f"     Warning: Could not cache haystack {hay_len} activations: {e}")
                
                # Make predictions
                # gpt2_val_hidden_acts shape: (n_configs, n_traces, T, n_embd)
                # W_opt shape: (5, 128)
                # Need to reshape for einsum: (n_configs*n_traces, T, n_embd) -> (n_configs*n_traces*T, n_embd)
                n_configs, n_traces, T_val, n_embd = gpt2_val_hidden_acts.shape
                gpt2_val_hidden_acts_flat = gpt2_val_hidden_acts.reshape(-1, n_embd)  # (n_configs*n_traces*T, n_embd)
                gpt2_val_preds_flat = gpt2_val_hidden_acts_flat @ W_opt.T  # (n_configs*n_traces*T, 5)
                gpt2_val_preds = gpt2_val_preds_flat.reshape(n_configs, n_traces, T_val, 5)  # (n_configs, n_traces, T, 5)
                
                # Compute MSE
                gpt2_sqerr_val_per_sample = np.sum((gpt2_val_preds - val_targets) ** 2, axis=-1)
                n_configs, n_traces, T_val = val_inputs.shape[:3]
                gpt2_sqerr_val_per_sample_reshaped = gpt2_sqerr_val_per_sample.reshape(n_configs, n_traces, T_val)
                gpt2_val_mask_reshaped = val_mask.reshape(n_configs, n_traces, T_val)
                gpt2_sqerr_val_per_sample_masked = gpt2_sqerr_val_per_sample_reshaped.copy()
                gpt2_sqerr_val_per_sample_masked[gpt2_val_mask_reshaped] = np.nan
                
                # Compute median MSE (traces first, then configs)
                gpt2_trace_medians_val = np.nanmedian(gpt2_sqerr_val_per_sample_masked, axis=1)
                gpt2_config_medians_val = np.nanmedian(gpt2_trace_medians_val, axis=1)
                gpt2_final_median_val = np.nanmedian(gpt2_config_medians_val)
                
                # Compute event errors
                val_data_for_events = {
                    'multi_sys_ys': val_inputs.reshape(n_configs, 1, n_traces, T_val, val_inputs.shape[-1])
                }
                gpt2_event_errors = compute_event_errors(gpt2_val_preds, val_targets, val_inputs, val_mask, None, None, val_data_for_events)
                
                print(f"    Haystack {hay_len}: MSE = {gpt2_final_median_val:.6f}")
                haystack_results[hay_len] = {
                    'validation_mse': gpt2_final_median_val,
                    'event_errors': gpt2_event_errors
                }
                
            except Exception as e:
                print(f"    ❌ Error testing haystack {hay_len}: {e}")
                haystack_results[hay_len] = None
        
        # Update total traces processed
        total_traces_processed += n_traces
        print(f"📊 Total traces processed so far: {total_traces_processed:,}")
        
        # Update plots and save event errors
        print(f"📊 Updating plots and saving event errors...")
        
        # Update main plots
        for hay_len in haystack_lengths:
            if hay_len in haystack_results and haystack_results[hay_len] is not None:
                plot_data = main_plots[hay_len]
                plot_data['data']['traces'].append(total_traces_processed)
                
                # Add event errors for each event type
                event_errors = haystack_results[hay_len]['event_errors']
                for i, event_type in enumerate(event_types):
                    plot_data['data']['gpt2_modified'][event_type].append(event_errors[i])
        
        # Save event errors to dictionary
        trace_key = f"{total_traces_processed:06d}"
        event_errors_dict[trace_key] = {}
        
        for hay_len in haystack_lengths:
            if hay_len in haystack_results and haystack_results[hay_len] is not None:
                event_errors_dict[trace_key][f"haystack_{hay_len}"] = {
                    'event_errors': haystack_results[hay_len]['event_errors'].tolist(),
                    'validation_mse': haystack_results[hay_len]['validation_mse']
                }
        
        # Save event errors to file
        event_errors_file = os.path.join(activations_cache_dir, "event_errors_by_traces.npz")
        try:
            np.savez_compressed(event_errors_file, event_errors_dict=event_errors_dict)
            print(f"    💾 Event errors saved for {total_traces_processed:,} traces")
        except Exception as e:
            print(f"    ❌ Error saving event errors: {e}")
        
        # Update plots
        for hay_len in haystack_lengths:
            plot_data = main_plots[hay_len]
            if plot_data['data']['traces']:  # Only update if we have data
                plt.figure(plot_data['fig'].number)
                plt.clf()  # Clear the figure
                
                # Recreate the plot
                plt.xlabel('Number of Training Traces')
                plt.ylabel('Event Error (MSE) - Log Scale')
                plt.title(f'Event Error Learning Curves: Haystack Length {hay_len}')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                
                # Plot each event type
                for i, event_type in enumerate(event_types):
                    if plot_data['data']['gpt2_modified'][event_type]:  # Only plot if we have data
                        plt.plot(
                            plot_data['data']['traces'],
                            plot_data['data']['gpt2_modified'][event_type],
                            color=colors[i],
                            label=f'GPT2-Modified {event_type}',
                            linewidth=2
                        )
                
                plt.legend()
                plt.tight_layout()
                
                # Save the plot
                plot_filename = f"learning_curves_haystack_{hay_len}.png"
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"    📊 Updated plot: {plot_filename}")
        
        # Clean up memory
        del batch_hidden_acts, targets, mask
        gc.collect()
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Final results:")
    print(f"   Total traces processed: {total_traces_processed:,}")
    print(f"   Final YA_T shape: {YA_T.shape}")
    print(f"   Final AA_T shape: {AA_T.shape}")
    print(f"   Final W_opt shape: {W_opt.shape}")

if __name__ == "__main__":
    main()
