import numpy as np
import pickle
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
import time
import glob


def load_and_convert_activations_to_float64(activations_array, chunk_size=500):
    """
    Convert large activations array to float64 in chunks to avoid memory overflow.
    
    Note: All activations including haystack activations are now cached and processed
    as float64 for maximum precision throughout all computations.
    """
    import gc
    import psutil
    import os
    
    if activations_array.dtype == np.float64:
        return activations_array
    
    print(f"    🔄 Converting activations to float64 in chunks (chunk_size={chunk_size})...")
    N, T, d = activations_array.shape
    print(f"    📊 Array shape: {activations_array.shape}, dtype: {activations_array.dtype}")
    print(f"    💾 Memory before conversion: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    
    result = np.zeros((N, T, d), dtype=np.float64)
    
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        result[i:end_idx] = activations_array[i:end_idx].astype(np.float64)
        
        # Force garbage collection every few chunks
        if i % (chunk_size * 5) == 0:
            gc.collect()
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            print(f"      Converted {i}/{N} samples... Memory: {memory_mb:.1f} MB")
    
    # Final garbage collection
    gc.collect()
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"    ✅ Conversion complete! Final memory: {memory_mb:.1f} MB")
    return result


# Add the TFs_do_KF_ICL directory to the Python path
sys.path.append('TFs_do_KF_ICL')
sys.path.append('TFs_do_KF_ICL/src')


from TFs_do_KF_ICL.src.models.lightning_base_model import BaseModel
from TFs_do_KF_ICL.src.models.gpt2 import GPT2
from TFs_do_KF_ICL.src.models.randomTransformer import RandomTransformer
from TFs_do_KF_ICL.src.core.config import Config


torch.manual_seed(42)
np.random.seed(42)




def load_previous_batch_results_from_traces(event_errors_file, haystack_lengths, main_plots, event_types, colors):
   """Load and plot all previous batch results from trace-based event errors file"""
   print(f"🔄 Loading previous batch results from trace-based file...")
   
   if not os.path.exists(event_errors_file):
       print(f"    ⚠️  No event errors file found, starting fresh...")
       return
   
   try:
       # Load event errors dictionary
       data = np.load(event_errors_file, allow_pickle=True)
       event_errors_dict = data['event_errors_dict'].item()
       
       print(f"    📊 Found event errors for {len(event_errors_dict)} trace counts")
       
       # Clear all existing plot data to start fresh
       print(f"🔄 Clearing existing plot data and starting fresh...")
       for hay_len in haystack_lengths:
           plot_data = main_plots[hay_len]
           plot_data['data']['traces'] = []
           for event_type in event_types:
               plot_data['data']['gpt2_pure'][event_type] = []
               plot_data['data']['gpt2_modified'][event_type] = []
       
       # Load and plot event errors for each trace count
       for trace_key in sorted(event_errors_dict.keys()):
           trace_count = int(trace_key)
           batch_num = trace_count // 20000  # Convert trace count to batch number
           
           print(f"    📊 Loading results for {trace_count:,} traces (batch {batch_num})...")
           
           # Create batch results structure for plotting
           batch_results_data = {
               'haystack_results': {}
           }
           
           # Add haystack results for this trace count
           for hay_len in haystack_lengths:
               haystack_key = f"haystack_{hay_len}"
               if haystack_key in event_errors_dict[trace_key]:
                   batch_results_data['haystack_results'][hay_len] = event_errors_dict[trace_key][haystack_key]
               else:
                   batch_results_data['haystack_results'][hay_len] = None
           
           # Update plots with this batch result
           update_main_plots(batch_num, {batch_num: batch_results_data}, haystack_lengths, main_plots, event_types, colors)
           print(f"    ✅ Loaded results for {trace_count:,} traces")
       
       print(f"✅ Loaded all previous batch results from trace-based file")
       
   except Exception as e:
       print(f"    ❌ Error loading event errors file: {e}")
       print(f"    🔄 Starting fresh...")


def update_main_plots(batch_num, batch_results, haystack_lengths, main_plots, event_types, colors):
   """Update the 3 main learning curve plots"""
   print(f"DEBUG: update_main_plots called with batch_num={batch_num}, batch_results keys={list(batch_results.keys())}")
   for hay_len in haystack_lengths:
           if batch_num in batch_results and 'haystack_results' in batch_results[batch_num] and hay_len in batch_results[batch_num]['haystack_results']:
               if batch_results[batch_num]['haystack_results'][hay_len] is not None:
                   plot_data = main_plots[hay_len]
                   traces = batch_num * 20000
                   
                   # Add new data point
                   plot_data['data']['traces'].append(traces)
                   
                   # Get event errors for this batch
                   event_errors = batch_results[batch_num]['haystack_results'][hay_len]['event_errors']
                   
                   # Get GPT2 Pure event errors from the stored baselines
                   if 'baseline_gpt2_pure' not in plot_data:
                       print(f"Warning: No baseline_gpt2_pure found for haystack {hay_len}, using zeros")
                       gpt2_pure_errors = [0.0] * len(event_types)
                   else:
                       gpt2_pure_errors = plot_data['baseline_gpt2_pure']
                   
                   # Update data for each event type
                   for i, event_type in enumerate(event_types):
                       # For GPT2 Pure, we want the same baseline value repeated
                       plot_data['data']['gpt2_pure'][event_type].append(gpt2_pure_errors[i])
                       plot_data['data']['gpt2_modified'][event_type].append(event_errors[i])
                   
                   print(f"DEBUG: Added data point - traces: {len(plot_data['data']['traces'])}, gpt2_pure: {len(plot_data['data']['gpt2_pure'][event_types[0]])}, gpt2_modified: {len(plot_data['data']['gpt2_modified'][event_types[0]])}")
               
               # Clear and redraw
               plt.figure(plot_data['fig'])
               plt.clf()
               plt.xlabel('Number of Training Traces')
               plt.ylabel('Event Error (MSE) - Log Scale')
               plt.title(f'Event Error Learning Curves: Haystack Length {hay_len}')
               plt.yscale('log')
               plt.grid(True, alpha=0.3)
               
               # Plot all curves
               for i, event_type in enumerate(event_types):
                   color = colors[i]
                   
                   # GPT2 Pure (flat horizontal line - same value for all x points)
                   if 'baseline_gpt2_pure' in plot_data and event_type in plot_data['baseline_gpt2_pure']:
                       # Create flat line that spans the data range
                       if len(plot_data['data']['traces']) > 0:
                           # Use actual data range
                           current_traces = plot_data['data']['traces']
                           x_min = min(current_traces)
                           x_max = max(current_traces)
                       else:
                           # Use default range when starting from scratch
                           x_min = 1000  # Start from 1k traces
                           x_max = 1000000  # Go up to 1M traces
                       
                       x_flat = [x_min, x_max]
                       # Use the baseline value directly
                       y_flat = [plot_data['baseline_gpt2_pure'][event_type]] * 2
                       
                       plt.plot(x_flat, y_flat, 
                                          color=color, linestyle='-', linewidth=2, alpha=0.7, marker=None,
                                          label=f'{event_type} (Pure)')
                   
                   # GPT2 Modified (connected curve) - only plot if there are data points
                   if len(plot_data['data']['traces']) > 0:
                       plt.plot(plot_data['data']['traces'], 
                                          plot_data['data']['gpt2_modified'][event_type], 
                                          color=color, linestyle='-', linewidth=2, marker='o', markersize=3,
                                          label=f'{event_type} (Modified)')
               
               plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
               plt.tight_layout()
               
               # Save updated plot
               plot_filename = f"TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k/main_learning_curves_haystack_{hay_len}_random.png"
               plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
               
               print(f"  📊 Updated haystack {hay_len} main plot: {traces:,} traces")
               
               # Print current event error values and improvements
               print(f"    Event Error Summary for {traces:,} traces:")
               for i, event_type in enumerate(event_types):
                   gpt2_pure_val = plot_data['data']['gpt2_pure'][event_type][-1]
                   gpt2_modified_val = plot_data['data']['gpt2_modified'][event_type][-1]
                   improvement = ((gpt2_pure_val - gpt2_modified_val) / gpt2_pure_val) * 100
                   print(f"      {event_type}: Pure={gpt2_pure_val:.6f}, Modified={gpt2_modified_val:.6f}, Improvement={improvement:+.1f}%")


def set_gpt2_pure_baselines(gpt2_pure_results, haystack_lengths, main_plots):
   """Set GPT2-Pure baseline values for all plots"""
   for hay_len in haystack_lengths:
       if hay_len in gpt2_pure_results and gpt2_pure_results[hay_len] is not None:
           plot_data = main_plots[hay_len]
           event_errors = gpt2_pure_results[hay_len]['event_errors']
           
           # Store the baseline values but don't add them to the arrays yet
           # They will be added when the first batch is processed
           plot_data['baseline_gpt2_pure'] = event_errors.copy()
           
           print(f"Set GPT2-Pure baselines for haystack {hay_len}")


def main():
   # Configuration
   config = Config()
   config.override("multi_sys_trace", True)
   config.override("max_sys_trace", 25)
   config.override("ny", 5)
   config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2))  # 57
   config.override("n_dims_out", 5)
   config.override("n_positions", 251)
  
   n_dims_in = config.n_dims_in   # 57
   n_dims_out = config.n_dims_out # 5
   n_positions = config.n_positions
   n_embd = 128
   n_layer = 12
   n_head = 8
   payload_flag_idx = n_dims_in - n_dims_out - 1  # 51
  
   device = "cpu"
  
   # Initialize recursive matrices with float64 precision
   YA_T = np.zeros((n_dims_out, n_embd), dtype=np.float64)  # 5 x 128
   AA_T = np.zeros((n_embd, n_embd), dtype=np.float64)     # 128 x 128
   
   # DEBUG: Print initial matrix dtypes
   print(f"🔍 DEBUG: Initial YA_T dtype: {YA_T.dtype}, shape: {YA_T.shape}")
   print(f"🔍 DEBUG: Initial AA_T dtype: {AA_T.dtype}, shape: {AA_T.shape}")
  
   # Initialize cached GPT2 activations to avoid recomputing
   cached_gpt2_activations = []
   cached_gpt2_targets = []
   cached_gpt2_masks = []
   total_traces_processed = 0
  
   # Initialize GPT2 Pure results (computed once, reused for all plots)
   gpt2_pure_results = {}
   gpt2_pure_computed = False
  
   # Initialize caching for haystack validation activations
   haystack_activations_cache = {}  # Key: haystack_length, Value: (activations, targets, masks)
  
   # Cache directory for activations
   cache_dir = "TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k"
   activations_cache_dir = os.path.join(cache_dir, "activations_cache")
   os.makedirs(activations_cache_dir, exist_ok=True)
  
   print(f"📁 Activations cache directory: {activations_cache_dir}")

   # Resume logic: Check for latest model cache file
   cache_dir = "TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k"
   # Check for existing matrices and event errors files
   matrices_file = os.path.join(cache_dir, "latest_matrices_random.npz")
   event_errors_file = os.path.join(cache_dir, "event_errors_by_traces_random.npz")
   last_completed_batch = 0
   
   if os.path.exists(matrices_file) and os.path.exists(event_errors_file):
       print(f"🔄 Found latest matrices file: {os.path.basename(matrices_file)}")
       print(f"🔄 Found event errors file: {os.path.basename(event_errors_file)}")
       
       try:
           # Load matrices
           cache_data = np.load(matrices_file)
           YA_T = cache_data['YA_T'].astype(np.float64)
           AA_T = cache_data['AA_T'].astype(np.float64)
           last_completed_batch = int(cache_data['batch_number'])
           total_traces_processed = int(cache_data['traces_processed'])
           print(f"   ✅ Loaded matrices: YA_T shape {YA_T.shape}, AA_T shape {AA_T.shape}")
           print(f"   📊 Last completed batch: {last_completed_batch}")
           print(f"   📊 Total traces processed: {total_traces_processed:,}")
           print(f"   🔍 DEBUG: Loaded YA_T dtype: {YA_T.dtype}, AA_T dtype: {AA_T.dtype}")
           
           # Load event errors to determine next batch
           event_data = np.load(event_errors_file, allow_pickle=True)
           event_dict = event_data['event_errors_dict'].item()
           completed_trace_counts = sorted([int(k) for k in event_dict.keys()])
           print(f"   📊 Completed trace counts: {completed_trace_counts}")
           
           # Calculate next batch to process
           if completed_trace_counts:
               last_completed_traces = max(completed_trace_counts)
               next_batch = (last_completed_traces // 20000) + 1
               print(f"   🔄 Last completed: {last_completed_traces:,} traces (batch {last_completed_traces // 20000})")
               print(f"   🔄 Resuming from batch {next_batch} ({next_batch * 20000:,} traces)")
               
               # Update batch tracking
               last_completed_batch = next_batch - 1
               total_traces_processed = last_completed_traces
           else:
               print(f"   ⚠️  No completed trace counts found, starting from scratch...")
               last_completed_batch = 0
               total_traces_processed = 0
           
           # Load previous batch results for plotting
           print(f"🔄 Loading previous batch results for plotting...")
           # Note: haystack_lengths, main_plots, event_types, colors will be defined later in main()
           # We'll skip plotting for now and let the main loop handle it
           
       except Exception as e:
           print(f"   ❌ Error loading files: {e}")
           print(f"   🔄 Starting from scratch...")
           last_completed_batch = 0
           total_traces_processed = 0
   else:
       print(f"🆕 No existing files found, starting from scratch...")


   # Load existing cached activations if they exist
   def load_existing_cache():
       """Load existing cached activations to avoid recomputation."""
       print("🔍 Checking for existing cached activations...")
      
       # Load training batch caches
       training_cache_files = glob.glob(os.path.join(activations_cache_dir, "training_batch_*_random.npz"))
       if training_cache_files:
           print(f"   📁 Found {len(training_cache_files)} cached training batches")
           for cache_file in sorted(training_cache_files):
               try:
                   cached_data = np.load(cache_file)
                   batch_num = int(cached_data['batch_number'])
                   print(f"       Batch {batch_num}: {cached_data['activations'].shape}")
               except Exception as e:
                   print(f"        Error reading {os.path.basename(cache_file)}: {e}")
      
       # Load haystack caches
       haystack_cache_files = glob.glob(os.path.join(activations_cache_dir, "haystack_*_validation_random.npz"))
       if haystack_cache_files:
           print(f"   📁 Found {len(haystack_cache_files)} cached haystack validations")
           for cache_file in sorted(haystack_cache_files):
               try:
                   cached_data = np.load(cache_file)
                   hay_len = int(cached_data['haystack_length'])
                   print(f"       Haystack {hay_len}: {cached_data['activations'].shape}")
               except Exception as e:
                   print(f"        Error reading {os.path.basename(cache_file)}: {e}")
      
       if not training_cache_files and not haystack_cache_files:
           print("   📝 No existing cache found - will compute all activations")
       print()
  
   # Skip loading cached activations for faster startup
   print("🔍 Skipping cached activations loading for faster startup...")
   
   # Initialize RandomTransformer model with random weights
   print(f"Initializing RandomTransformer model with random weights...")
   
   gpt2_model = RandomTransformer(
       n_dims_in=n_dims_in,
       n_positions=n_positions,
       n_embd=n_embd,
       n_layer=n_layer,
       n_head=n_head,
       n_dims_out=n_dims_out
   )
   print("Successfully initialized RandomTransformer model with random weights")
   
   # Note: RandomTransformer freezes backbone and _read_in layers internally
   # and only allows gradients for the _read_out layer

   gpt2_model = gpt2_model.to(device)
   
   # Convert model to double precision for float64 computations
   print("Converting RandomTransformer model to double precision...")
   gpt2_model = gpt2_model.double()
   
   # Initialize 3 main learning curve plots (one per haystack)
   print("\nCreating 3 main learning curve plots (one per haystack)...")
   haystack_lengths = [1, 2, 5]
   
   # Define event types and colors (10 total: 5 after initial + 5 after final)
   event_types = [f"{k}_after_initial" for k in [1,2,3,7,8]] + [f"{k}_after_final" for k in [1,2,3,7,8]]
   colors = ['red', 'blue', 'green', 'orange', 'purple', 'red', 'blue', 'green', 'orange', 'purple']
   
   # Create 3 main plots
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
               'gpt2_pure': {event: [] for event in event_types},
               'gpt2_modified': {event: [] for event in event_types}
           }
       }
      print(f"Created main plot for haystack length {hay_len}")
   
   # Load previous batch results for plotting if resuming
   if last_completed_batch > 0:
       print(f"🔄 Loading previous batch results for plotting...")
       load_previous_batch_results_from_traces(event_errors_file, haystack_lengths, main_plots, event_types, colors)







   # Main loop: process existing 20k batch files
   batch_dir = "TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k"
   batch_files = glob.glob(f"{batch_dir}/batch_*.pkl")
  
   # Sort batch files by batch number to ensure correct order (1, 2, 3, ..., 50)
   def extract_batch_number(filename):
       basename = os.path.basename(filename)
       # Extract number from "batch_XX_20k_traces.pkl" format
       parts = basename.split('_')
       if len(parts) >= 2:
           try:
               return int(parts[1])
           except ValueError:
               return 0
       return 0
  
   batch_files = sorted(batch_files, key=extract_batch_number)
  
   if not batch_files:
       print(f"❌ No batch files found in {batch_dir}")
       return
  
   num_batches = len(batch_files)
   total_traces = num_batches * 20000  # Each batch has 20k traces
  
   # Determine starting batch based on resume logic
   start_batch_idx = last_completed_batch
   remaining_batches = batch_files[start_batch_idx:]
   remaining_traces = len(remaining_batches) * 20000
  
   # Verify the order
   print("📋 Batch processing order:")
   for i, batch_file in enumerate(remaining_batches[:5]):  # Show first 5 remaining
       print(f"   {i+1+start_batch_idx}: {os.path.basename(batch_file)}")
   if len(remaining_batches) > 5:
       print(f"   ... ({len(remaining_batches)-5} more batches)")
   print()
  
   # Store results for each batch
   batch_results = {}
  
   print(f"\n" + "="*60)
   print(f"PROCESSING {len(remaining_batches)} REMAINING BATCH FILES ({remaining_traces:,} ADDITIONAL TRACES)")
   print("="*60)
   print("📊 Using CUMULATIVE processing: Each batch adds to previous activations")
   if start_batch_idx > 0:
       print(f"🔄 Resuming from batch {start_batch_idx + 1} with {total_traces_processed:,} existing traces")
   print(f"🔄 Batch {start_batch_idx + 1}: {total_traces_processed + 20000:,} traces → Batch {start_batch_idx + 2}: {total_traces_processed + 40000:,} traces...")
   print("="*60)
  
   for batch_idx, batch_file in enumerate(remaining_batches):
       # Update batch index to account for skipped batches
       actual_batch_idx = batch_idx + start_batch_idx
           
       print(f"\n" + "="*50)
       print(f"BATCH {actual_batch_idx + 1}/{num_batches}: Processing {os.path.basename(batch_file)}")
       print("="*50)
      
       # Load the existing 20k traces from this batch file
       try:
           with open(batch_file, 'rb') as f:
               batch_data = pickle.load(f)
          
           if 'multi_sys_ys' in batch_data:
               batch_inputs = batch_data['multi_sys_ys']  # Shape: (1, 20000, 1, 251, 57)
               batch_inputs = batch_inputs[0, :, 0, :, :]  # Reshape to (20000, 251, 57)
               print(f" Successfully loaded batch with shape: {batch_inputs.shape}")
           else:
               print(f"❌ No 'multi_sys_ys' found in {batch_file}")
               continue
              
       except Exception as e:
           print(f"❌ Error loading batch file {batch_file}: {e}")
           continue
      
       # Get payloads and create targets
       batch_payloads = batch_inputs[..., -n_dims_out:]
       batch_targets = np.zeros_like(batch_payloads, dtype=np.float64)
       batch_targets[:, :-1, :] = batch_payloads[:, 1:, :].astype(np.float64)  # shift left by 1
      
       # Create training mask
       batch_mask = np.ones((batch_inputs.shape[0], batch_inputs.shape[1]), dtype=bool)
       batch_mask[:, :-1] = (batch_inputs[:, 1:, payload_flag_idx] == 0)
       batch_mask[:, -1] = True
      
       # Get GPT2 activations for this batch
       print(f"Computing GPT2 activations for batch {actual_batch_idx + 1}...")
      
       # Check if activations are already cached
       batch_cache_file = os.path.join(activations_cache_dir, f"training_batch_{actual_batch_idx + 1:02d}_random.npz")
      
       if os.path.exists(batch_cache_file):
           print(f"    📁 Loading cached activations from: {os.path.basename(batch_cache_file)}")
           try:
               cached_data = np.load(batch_cache_file)
               batch_hidden_acts = cached_data['activations'].astype(np.float64)
               print(f"     Successfully loaded cached activations with shape: {batch_hidden_acts.shape}")
           except Exception as e:
               print(f"      Error loading cached activations: {e}")
               print(f"    🔄 Computing activations from scratch...")
               batch_hidden_acts = None
       else:
           batch_hidden_acts = None
      
       # Compute activations if not cached
       if batch_hidden_acts is None:
           print(f"    🔄 Computing activations from scratch...")
           batch_hidden_acts = []
           gpt2_model.eval()
           try:
               with torch.no_grad():
                   for i in range(len(batch_inputs)):
                       try:
                           x = torch.tensor(batch_inputs[i:i+1], dtype=torch.float64).to(device)
                           embeds = gpt2_model._read_in(x)
                           hidden = gpt2_model._backbone(inputs_embeds=embeds).last_hidden_state
                           batch_hidden_acts.append(hidden.cpu().numpy().astype(np.float64))
                       except Exception as e:
                           print(f"          Error processing trace {i}: {e}")
                           continue
              
                   if batch_hidden_acts:
                       batch_hidden_acts = np.concatenate(batch_hidden_acts, axis=0).astype(np.float64)
                       print(f"         Successfully computed activations for {len(batch_hidden_acts)} traces")
                       
                       # Cache the activations for future use
                       print(f"        💾 Caching activations to: {os.path.basename(batch_cache_file)}")
                       try:
                           np.savez_compressed(
                               batch_cache_file,
                               activations=batch_hidden_acts,
                               targets=batch_targets,
                               masks=batch_mask,
                               batch_number=actual_batch_idx + 1,
                               timestamp=time.time()
                           )
                           print(f"         Activations cached successfully")
                       except Exception as e:
                           print(f"          Warning: Could not cache activations: {e}")
                   else:
                       print(f"        ❌ No valid activations computed for this batch")
                       continue
           except Exception as e:
               print(f"        ❌ Error computing GPT2 activations: {e}")
               continue
      
       # Cache the activations, targets, and masks for this batch
       cached_gpt2_activations.append(batch_hidden_acts)
       cached_gpt2_targets.append(batch_targets)
       cached_gpt2_masks.append(batch_mask)
       total_traces_processed = (actual_batch_idx + 1) * 20000
      
       print(f"    📊 Total traces cached so far: {total_traces_processed:,}")
      
       # Load GPT2 Pure results from original file (since we're using random initialization)
       if not gpt2_pure_computed:
           print(f"\n🔬 Loading GPT2 Pure baseline from original file...")
           try:
               with open('gpt2_pure_event_errors.pkl', 'rb') as f:
                   gpt2_pure_event_errors_dict = pickle.load(f)
               print(f"    ✅ Loaded GPT2 Pure event errors from original file")
               
               # Convert back to the expected format
               for hay_len in haystack_lengths:
                   if hay_len in gpt2_pure_event_errors_dict:
                       gpt2_pure_results[hay_len] = {
                           'event_errors': np.array(gpt2_pure_event_errors_dict[hay_len], dtype=np.float64)
                       }
               gpt2_pure_computed = True
               print(f"    📊 GPT2 Pure event errors loaded for haystacks: {list(gpt2_pure_results.keys())}")
           except FileNotFoundError:
               print(f"    ⚠️  Original GPT2 Pure event errors file not found, will compute from scratch")
               # Fall back to computing from scratch if file not found
               gpt2_pure_computed = False
           except Exception as e:
               print(f"    ❌ Error loading GPT2 Pure event errors: {e}")
               gpt2_pure_computed = False
       
       # Compute GPT2 Pure results from scratch if not loaded
       if not gpt2_pure_computed:
           #     print(f"\n🔬 Computing GPT2 Pure baseline for all haystack lengths (one-time computation)...")
           #    
           #     # Load fresh GPT2 model for pure computation
           #     gpt2_pure_model = GPT2.load_from_checkpoint(
           #         gpt2_checkpoint_path,
           #         n_dims_in=n_dims_in,
           #         n_positions=n_positions,
           #         n_embd=n_embd,
           #         n_layer=n_layer,
           #         n_head=n_head,
           #         n_dims_out=n_dims_out
           #     )
           #     gpt2_pure_model = gpt2_pure_model.to(device)
           #     gpt2_pure_model = gpt2_pure_model.double()
           #    
           #     # Process all haystack lengths at once
           #     haystack_lengths_pure = [1, 2, 5]
           #     for hay_len in haystack_lengths_pure:
           #         print(f"  Testing GPT2 Pure on haystack length {hay_len}...")
           #        
           #         # Check if we have cached activations for this haystack
           #         haystack_cache_file = os.path.join(activations_cache_dir, f"haystack_{hay_len}_validation.npz")
           #        
           #         if os.path.exists(haystack_cache_file):
           #             print(f"    📁 Using cached activations for haystack {hay_len}")
           #             try:
           #                 cached_data = np.load(haystack_cache_file)
           #                 val_inputs = cached_data['inputs']
           #                 val_targets = cached_data['targets']
           #                 val_mask = cached_data['masks']
           #                 gpt2_pure_val_hidden_acts = cached_data['activations']
           #                
           #                 # Convert haystack activations to float64 for precision
           #                 gpt2_pure_val_hidden_acts = gpt2_pure_val_hidden_acts.astype(np.float64)
           #                 print(f"    📊 GPT2 Pure haystack activations: {gpt2_pure_val_hidden_acts.dtype}, shape: {gpt2_pure_val_hidden_acts.shape}")
           #                
           #                 print(f"     Loaded cached data: inputs {val_inputs.shape}, targets {val_targets.shape}, activations {gpt2_pure_val_hidden_acts.shape}")
           #                
           #                 # Get predictions using original GPT2 model (no RLS optimization) - use cached activations
           #                 gpt2_pure_val_preds = []
           #                 gpt2_pure_model.eval()
           #                 with torch.no_grad():
           #                     for i in range(len(val_inputs)):
           #                         x = torch.tensor(val_inputs[i:i+1], dtype=torch.float64).to(device)
           #                         embeds = gpt2_pure_model._read_in(x)
           #                         # Use the cached activations instead of recomputing
           #                         hidden = torch.tensor(gpt2_pure_val_hidden_acts[i:i+1], dtype=torch.float64).to(device)
           #                         pred = gpt2_pure_model._read_out(hidden)  # Shape: (1, T, 5)
           #                         gpt2_pure_val_preds.append(pred.cpu().numpy().astype(np.float64))
           #                
           #                 gpt2_pure_val_preds = np.concatenate(gpt2_pure_val_preds, axis=0).astype(np.float64)  # Shape: (N_val, T_val, 5)
           #                
           #                 # Compute MSE
           #                 gpt2_pure_sqerr_val_per_sample = np.sum((gpt2_pure_val_preds - val_targets) ** 2, axis=-1)
           #                 # Fix the reshape to use val_inputs instead of val_data
           #                 N_val, T_val = val_inputs.shape[:2]
           #                 gpt2_pure_sqerr_val_per_sample_reshaped = gpt2_pure_sqerr_val_per_sample.reshape(N_val, T_val)
           #                 gpt2_pure_val_mask_reshaped = val_mask.reshape(N_val, T_val)
           #                 gpt2_pure_sqerr_val_per_sample_masked = gpt2_pure_sqerr_val_per_sample_reshaped.copy()
           #                 gpt2_pure_sqerr_val_per_sample_masked[gpt2_pure_val_mask_reshaped] = np.nan
           #                
           #                 # Compute time-series MSE: median across traces, then across configs for each time step
           #                 gpt2_pure_mse_over_time = np.nanmedian(gpt2_pure_sqerr_val_per_sample_masked, axis=0)  # Shape: (T_val,)
           #                
           #                 gpt2_pure_trace_medians_val = np.nanmedian(gpt2_pure_sqerr_val_per_sample_masked, axis=0)
           #                 gpt2_pure_final_median_val = np.nanmedian(gpt2_pure_trace_medians_val)
           #                
           #                 # Compute event errors (this is what we really need for the plots)
           #                 # Create minimal val_data structure for compute_event_errors
           #                 N_val, T_val = val_inputs.shape[:2]
           #                 n_configs = 1  # Assume single config for cached data
           #                 n_traces = N_val
           #                 minimal_val_data = {
           #                     'multi_sys_ys': val_inputs.reshape(n_configs, 1, n_traces, T_val, val_inputs.shape[-1])
           #                 }
           #                
           #                 gpt2_pure_event_errors = compute_event_errors(gpt2_pure_val_preds, val_targets, val_inputs, val_mask, config, n_dims_in, minimal_val_data)
           #                
           #                 gpt2_pure_results[hay_len] = {
           #                     'validation_mse': gpt2_pure_final_median_val,
           #                     'event_errors': gpt2_pure_event_errors,
           #                     'mse_over_time': gpt2_pure_mse_over_time
           #                 }
           #                
           #                 print(f"     Haystack {hay_len}: MSE = {gpt2_pure_final_median_val:.6f}, Event Errors computed (using cached activations)")
           #                
           #                 # Cache in memory for future use
           #                 haystack_activations_cache[hay_len] = (val_inputs, val_targets, val_mask)
           #                 haystack_activations_cache[f"{hay_len}_activations"] = gpt2_pure_val_hidden_acts.copy()
           #                
           #             except Exception as e:
           #                 print(f"    ❌ Error using cached activations for haystack {hay_len}: {e}")
           #                 print(f"    🔄 Falling back to computing from scratch...")
           #                 gpt2_pure_results[hay_len] = None
           #                 continue
                      
           #         else:
           #             print(f"    🔄 No cache found, computing GPT2 Pure activations from scratch...")
           #            
           #             # Load validation data for this haystack length
           #             val_data_path = f'TFs_do_KF_ICL/src/DataRandomTransformer/val_interleaved_traces_ortho_haar_ident_C_haystack_len_{hay_len}.pkl'
           #             try:
           #                 with open(val_data_path, 'rb') as f:
           #                     val_data = pickle.load(f)
           #                
           #                 val_inputs = val_data['multi_sys_ys']  # (n_configs,1,n_traces,T,D)
           #                 N_val = np.prod(val_inputs.shape[:-2])
           #                 T_val = val_inputs.shape[-2]
           #                 val_inputs = val_inputs.reshape(N_val, T_val, n_dims_in)
           #                
           #                 # Get payloads and create targets
           #                 val_payloads = val_inputs[..., -n_dims_out:]
           #                 val_targets = np.zeros_like(val_payloads, dtype=np.float64)
           #                 val_targets[:, :-1, :] = val_payloads[:, 1:, :].astype(np.float64)
           #                
           #                 # Create validation mask
           #                 val_mask = np.ones((val_inputs.shape[0], val_inputs.shape[1]), dtype=bool)
           #                 val_mask[:, :-1] = (val_inputs[:, 1:, payload_flag_idx] == 0)
           #                 val_mask[:, -1] = True
                      
           # This section is commented out since gpt2_pure_model is not defined
           # The GPT2 Pure computation is handled by loading from the original file above
           pass
      
       # Get current batch data for recursive updates
       print(f"    🔄 Processing current batch for recursive update...")
       
       # Load current batch's cached activations
       current_batch_cache_file = os.path.join(activations_cache_dir, f"training_batch_{actual_batch_idx + 1:02d}_random.npz")
       if os.path.exists(current_batch_cache_file):
           print(f"    📁 Loading cached activations from: {os.path.basename(current_batch_cache_file)}")
           cached_data = np.load(current_batch_cache_file)
           current_batch_activations = cached_data['activations'].astype(np.float64)
           current_batch_targets = cached_data['targets'].astype(np.float64)
           current_batch_masks = cached_data['masks'].astype(bool)
           print(f"    📊 Loaded current batch shape: {current_batch_activations.shape}")
       else:
           print(f"    ❌ No cached activations found for batch {actual_batch_idx + 1}")
           continue
      
       print(f"    📈 Current batch shape: {current_batch_activations.shape}")
      
       # Zero out masked hidden states for current batch
       current_batch_activations_masked = current_batch_activations.copy()
       current_batch_activations_masked[current_batch_masks] = 0
      
       # Reshape for matrix operations using current batch data
       N_current, T_total, d = current_batch_activations_masked.shape
       A_current = current_batch_activations_masked.transpose(2, 0, 1).reshape(d, N_current * T_total).astype(np.float64)  # 128 x (N*T)
       Y_current = current_batch_targets.transpose(2, 0, 1).reshape(n_dims_out, N_current * T_total).astype(np.float64)   # 5 x (N*T)
       
       # DEBUG: Print input matrix dtypes
       print(f"🔍 DEBUG: A_current dtype: {A_current.dtype}, shape: {A_current.shape}")
       print(f"🔍 DEBUG: Y_current dtype: {Y_current.dtype}, shape: {Y_current.shape}")
       print(f"🔍 DEBUG: current_batch_activations_masked dtype: {current_batch_activations_masked.dtype}")
       print(f"🔍 DEBUG: current_batch_targets dtype: {current_batch_targets.dtype}")
      
       # Apply mask to get only unmasked positions
       unmasked_indices = ~current_batch_masks.flatten()
       A_current_unmasked = A_current[:, unmasked_indices].astype(np.float64)  # 128 x num_unmasked
       Y_current_unmasked = Y_current[:, unmasked_indices].astype(np.float64)  # 5 x num_unmasked
       
       # DEBUG: Print unmasked matrix dtypes
       print(f"🔍 DEBUG: A_current_unmasked dtype: {A_current_unmasked.dtype}, shape: {A_current_unmasked.shape}")
       print(f"🔍 DEBUG: Y_current_unmasked dtype: {Y_current_unmasked.dtype}, shape: {Y_current_unmasked.shape}")
      
       # Recursive updates using current batch data
       print(f"Updating recursive matrices with current batch data...")
      
       # For the first batch, initialize the matrices
       if batch_idx == 0:
           # Process in chunks to avoid memory spikes
           chunk_size = 10000  # Process 10k columns at a time
           num_chunks = (A_current_unmasked.shape[1] + chunk_size - 1) // chunk_size
           
           print(f"    🔄 Processing matrix products in {num_chunks} chunks of {chunk_size} columns...")
           
           # Initialize accumulation matrices
           matrix_product_YA = np.zeros((5, 128), dtype=np.float64)
           matrix_product_AA = np.zeros((128, 128), dtype=np.float64)
           
           for chunk_idx in range(num_chunks):
               start_idx = chunk_idx * chunk_size
               end_idx = min(start_idx + chunk_size, A_current_unmasked.shape[1])
               
               # Get chunk
               A_chunk = A_current_unmasked[:, start_idx:end_idx]  # (128, chunk_size)
               Y_chunk = Y_current_unmasked[:, start_idx:end_idx]  # (5, chunk_size)
               
               # Compute chunk products
               chunk_YA = Y_chunk @ A_chunk.T  # (5, 128)
               chunk_AA = A_chunk @ A_chunk.T  # (128, 128)
               
               # Accumulate
               matrix_product_YA += chunk_YA
               matrix_product_AA += chunk_AA
               
               # Memory cleanup
               del A_chunk, Y_chunk, chunk_YA, chunk_AA
               
               if chunk_idx % 10 == 0:
                   print(f"      Processed chunk {chunk_idx + 1}/{num_chunks}")
           
           print(f"🔍 DEBUG: matrix_product_YA dtype: {matrix_product_YA.dtype}, shape: {matrix_product_YA.shape}")
           print(f"🔍 DEBUG: matrix_product_AA dtype: {matrix_product_AA.dtype}, shape: {matrix_product_AA.shape}")
           
           # Convert to float64 for recursive accumulation
           YA_T = matrix_product_YA.astype(np.float64)  # 5 x 128
           AA_T = matrix_product_AA.astype(np.float64)  # 128 x 128
           
           # Clean up large intermediate arrays
           del matrix_product_YA, matrix_product_AA
           import gc
           gc.collect()
           
           # DEBUG: Print YA_T and AA_T dtypes after first assignment
           print(f"🔍 DEBUG: After first assignment - YA_T dtype: {YA_T.dtype}, AA_T dtype: {AA_T.dtype}")
       else:
           # Process in chunks to avoid memory spikes
           chunk_size = 10000  # Process 10k columns at a time
           num_chunks = (A_current_unmasked.shape[1] + chunk_size - 1) // chunk_size
           
           print(f"    🔄 Processing matrix products in {num_chunks} chunks of {chunk_size} columns...")
           
           # Initialize accumulation matrices
           matrix_product_YA = np.zeros((5, 128), dtype=np.float64)
           matrix_product_AA = np.zeros((128, 128), dtype=np.float64)
           
           for chunk_idx in range(num_chunks):
               start_idx = chunk_idx * chunk_size
               end_idx = min(start_idx + chunk_size, A_current_unmasked.shape[1])
               
               # Get chunk
               A_chunk = A_current_unmasked[:, start_idx:end_idx]  # (128, chunk_size)
               Y_chunk = Y_current_unmasked[:, start_idx:end_idx]  # (5, chunk_size)
               
               # Compute chunk products
               chunk_YA = Y_chunk @ A_chunk.T  # (5, 128)
               chunk_AA = A_chunk @ A_chunk.T  # (128, 128)
               
               # Accumulate
               matrix_product_YA += chunk_YA
               matrix_product_AA += chunk_AA
               
               # Memory cleanup
               del A_chunk, Y_chunk, chunk_YA, chunk_AA
               
               if chunk_idx % 10 == 0:
                   print(f"      Processed chunk {chunk_idx + 1}/{num_chunks}")
           
           print(f"🔍 DEBUG: matrix_product_YA dtype: {matrix_product_YA.dtype}, shape: {matrix_product_YA.shape}")
           print(f"🔍 DEBUG: matrix_product_AA dtype: {matrix_product_AA.dtype}, shape: {matrix_product_AA.shape}")
           print(f"🔍 DEBUG: Before addition - YA_T dtype: {YA_T.dtype}, AA_T dtype: {AA_T.dtype}")
           
           # For subsequent batches, add the contribution from the current batch (convert to float64)
           YA_T += matrix_product_YA.astype(np.float64)  # 5 x 128
           AA_T += matrix_product_AA.astype(np.float64)  # 128 x 128
           
           # Clean up large intermediate arrays
           del matrix_product_YA, matrix_product_AA
           import gc
           gc.collect()
           
           # DEBUG: Print YA_T and AA_T dtypes after addition
           print(f"🔍 DEBUG: After addition - YA_T dtype: {YA_T.dtype}, AA_T dtype: {AA_T.dtype}")
          
           print(f"    ➕ Added current batch contribution: {N_current * T_total} positions, {np.sum(~current_batch_masks.flatten())} unmasked")
           print(f"    📊 Running totals: YA_T norm = {np.linalg.norm(YA_T):.6f}, AA_T norm = {np.linalg.norm(AA_T):.6f}")
      
       print(f"Batch {actual_batch_idx + 1} complete. YA_T shape: {YA_T.shape}, AA_T shape: {AA_T.shape}")
       print(f"    📊 Using current batch data: {total_traces_processed:,} traces, {N_current * T_total:,} current positions")
       print(f"    🔍 Matrix diagnostics: YA_T range [{YA_T.min():.6f}, {YA_T.max():.6f}], AA_T range [{AA_T.min():.6f}, {AA_T.max():.6f}]")
      
       # Compute W_opt for current batch
       print(f"Computing W_opt for batch {actual_batch_idx + 1}...")
       
       # DEBUG: Print final matrix dtypes before inversion
       print(f"🔍 DEBUG: Final YA_T dtype: {YA_T.dtype}, shape: {YA_T.shape}")
       print(f"🔍 DEBUG: Final AA_T dtype: {AA_T.dtype}, shape: {AA_T.shape}")
       
       try:
           # Try regular inverse first (both matrices are already float64)
           W_opt_current = YA_T @ np.linalg.inv(AA_T)
           print(" Used regular matrix inverse")
       except np.linalg.LinAlgError:
           # Use pseudoinverse if matrix is singular (both matrices are already float64)
           W_opt_current = YA_T @ np.linalg.pinv(AA_T)
           print("  Matrix was singular - used pseudoinverse")
       except Exception as e:
           print(f"❌ Error computing W_opt: {e}")
           print("Continuing with next batch...")
           continue
      
       print(f"Current W_opt shape: {W_opt_current.shape}")
       print(f"🔍 DEBUG: W_opt_current dtype: {W_opt_current.dtype}")
      
       # Test current W_opt on haystacks
       print(f"Testing current W_opt on haystacks...")
       haystack_lengths = [1, 2, 5]
       batch_haystack_results = {}
      
       for hay_len in haystack_lengths:
           print(f"  Testing haystack length {hay_len}...")
          
           try:
               # Check if haystack activations are already cached
               haystack_cache_file = os.path.join(activations_cache_dir, f"haystack_{hay_len}_validation_random.npz")
              
               if os.path.exists(haystack_cache_file):
                   print(f"    📁 Using cached haystack {hay_len} activations")
                   try:
                       cached_data = np.load(haystack_cache_file)
                       val_inputs = cached_data['inputs']
                       val_targets = cached_data['targets']
                       val_mask = cached_data['masks']
                       
                       # Load haystack activations and convert to float64 for precision
                       print(f"    🔄 Loading haystack activations and converting to float64...")
                       gpt2_val_hidden_acts = cached_data['activations'].astype(np.float64)
                       print(f"    📊 Haystack activations: {gpt2_val_hidden_acts.dtype}, shape: {gpt2_val_hidden_acts.shape}")
                       
                       # Zero out masked hidden states
                       gpt2_val_hidden_acts[val_mask] = 0
                     
                       # Make predictions using current W_opt (GPT2-Pseudoinverse)
                       # Only handle 4D cached format: (n_configs, n_traces, T, n_embd)
                       n_configs, n_traces, T_val, n_embd = gpt2_val_hidden_acts.shape
                       # Reshape for einsum: (n_configs, n_traces, T, n_embd) -> (n_configs*n_traces, T, n_embd)
                       gpt2_val_hidden_acts_flat = gpt2_val_hidden_acts.reshape(-1, T_val, n_embd)
                       gpt2_val_preds_flat = np.einsum('od,ntd->nto', W_opt_current, gpt2_val_hidden_acts_flat)
                       # Reshape back to (n_configs, n_traces, T, n_dims_out)
                       gpt2_val_preds = gpt2_val_preds_flat.reshape(n_configs, n_traces, T_val, n_dims_out)
                       
                       # Compute MSE for GPT2-Pseudoinverse
                       gpt2_sqerr_val_per_sample = np.sum((gpt2_val_preds - val_targets) ** 2, axis=-1)  # (n_configs, n_traces, T)
                       
                       # Apply mask
                       gpt2_sqerr_val_per_sample_masked = gpt2_sqerr_val_per_sample.copy()
                       gpt2_sqerr_val_per_sample_masked[val_mask] = np.nan
                       
                       # Compute time-series MSE: median across traces first, then across configs for each time step
                       gpt2_trace_medians_val = np.nanmedian(gpt2_sqerr_val_per_sample_masked, axis=1)  # (n_configs, T)
                       gpt2_mse_over_time = np.nanmedian(gpt2_trace_medians_val, axis=0)  # Shape: (T_val,)
                       gpt2_config_medians_val = np.nanmedian(gpt2_trace_medians_val, axis=1)  # (n_configs,)
                       gpt2_final_median_val = np.nanmedian(gpt2_config_medians_val)
                       
                       # Reconstruct val_data structure for 4D format
                       reconstructed_val_data = {
                           'multi_sys_ys': val_inputs.reshape(n_configs, 1, n_traces, T_val, val_inputs.shape[-1])
                       }
                       
                       #print(gpt2_val_preds.shape, val_targets.shape, val_inputs.shape, val_mask.shape, config, n_dims_in, reconstructed_val_data)
                       gpt2_event_errors = compute_event_errors(gpt2_val_preds, val_targets, val_inputs, val_mask, config, n_dims_in, reconstructed_val_data)
                      
                       # Store GPT2-Pseudoinverse results
                       batch_haystack_results[hay_len] = {
                           'validation_mse': gpt2_final_median_val,
                           'event_errors': gpt2_event_errors,
                           'mse_over_time': gpt2_mse_over_time
                       }
                      
                       print(f"     Haystack {hay_len}: MSE = {gpt2_final_median_val:.6f}, Event Errors computed (using cached activations)")
                      
                       # Cache in memory for future use
                       haystack_activations_cache[hay_len] = (val_inputs, val_targets, val_mask)
                       haystack_activations_cache[f"{hay_len}_activations"] = gpt2_val_hidden_acts.copy()
                       
                       # Free memory immediately after processing
                       del gpt2_val_hidden_acts, gpt2_val_preds, gpt2_sqerr_val_per_sample
                       import gc
                       gc.collect()
                      
                   except Exception as e:
                       print(f"    ❌ Error using cached activations for haystack {hay_len}: {e}")
                       print(f"    🔄 Falling back to computing from scratch...")
                       batch_haystack_results[hay_len] = None
                       continue
                      
               else:
                   print(f"    🔄 No cache found, computing haystack activations from scratch...")
                  
                   # Load validation data for this haystack length
                   val_data_path = f'TFs_do_KF_ICL/src/DataRandomTransformer/val_interleaved_traces_ortho_haar_ident_C_haystack_len_{hay_len}.pkl'
                   try:
                       with open(val_data_path, 'rb') as f:
                           val_data = pickle.load(f)
                      
                       val_inputs = val_data['multi_sys_ys']  # (n_configs,1,n_traces,T,D)
                       n_configs, _, n_traces, T_val, n_dims_in = val_inputs.shape
                       # Keep the original shape: (n_configs, n_traces, T, D)
                       val_inputs = val_inputs.reshape(n_configs, n_traces, T_val, n_dims_in)
                      
                       # Get payloads and create targets
                       val_payloads = val_inputs[..., -n_dims_out:]
                       val_targets = np.zeros_like(val_payloads, dtype=np.float64)
                       val_targets[:, :, :-1, :] = val_payloads[:, :, 1:, :].astype(np.float64)
                      
                       # Create validation mask
                       val_mask = np.ones((n_configs, n_traces, T_val), dtype=bool)
                       val_mask[:, :, :-1] = (val_inputs[:, :, 1:, payload_flag_idx] == 0)
                       val_mask[:, :, -1] = True
                      
                       # Get GPT2 activations for validation (using original model)
                       gpt2_val_hidden_acts = []
                       gpt2_model.eval()
                       with torch.no_grad():
                           for config_i in range(n_configs):
                               config_activations = []
                               for trace_i in range(n_traces):
                                   x = torch.tensor(val_inputs[config_i, trace_i:trace_i+1], dtype=torch.float64).to(device)
                                   embeds = gpt2_model._read_in(x)
                                   hidden = gpt2_model._backbone(inputs_embeds=embeds).last_hidden_state
                                   config_activations.append(hidden.cpu().numpy().astype(np.float64))
                               gpt2_val_hidden_acts.append(np.concatenate(config_activations, axis=0))
                       
                       # Stack to get shape (n_configs, n_traces, T, n_embd)
                       gpt2_val_hidden_acts = np.stack(gpt2_val_hidden_acts, axis=0).astype(np.float64)
                      
                       # Cache the haystack data for future use
                       haystack_activations_cache[hay_len] = (val_inputs, val_targets, val_mask)
                       haystack_activations_cache[f"{hay_len}_activations"] = gpt2_val_hidden_acts.copy()
                      
                       # Save to disk cache as float64 for precision
                       print(f"    💾 Caching haystack {hay_len} activations as float64...")
                       try:
                           np.savez_compressed(
                               haystack_cache_file,
                               inputs=val_inputs,
                               targets=val_targets,
                               masks=val_mask,
                               activations=gpt2_val_hidden_acts.astype(np.float64),  # Save as float64
                               haystack_length=hay_len,
                               timestamp=time.time()
                           )
                           print(f"     Haystack {hay_len} activations cached")
                       except Exception as e:
                           print(f"      Warning: Could not cache haystack {hay_len} activations: {e}")
                  
                   except Exception as e:
                       print(f"        ❌ Error loading haystack {hay_len} data: {e}")
                       print(f"        Continuing with next haystack length...")
                       batch_haystack_results[hay_len] = None
                       continue
                  
                   # Zero out masked hidden states
                   gpt2_val_hidden_acts[val_mask] = 0
                  
                   # Make predictions using current W_opt
                   # Reshape for einsum: (n_configs, n_traces, T, n_embd) -> (n_configs*n_traces, T, n_embd)
                   n_configs, n_traces, T_val, n_embd = gpt2_val_hidden_acts.shape
                   gpt2_val_hidden_acts_flat = gpt2_val_hidden_acts.reshape(-1, T_val, n_embd)
                   gpt2_val_preds_flat = np.einsum('od,ntd->nto', W_opt_current, gpt2_val_hidden_acts_flat)
                   # Reshape back to (n_configs, n_traces, T, n_dims_out)
                   gpt2_val_preds = gpt2_val_preds_flat.reshape(n_configs, n_traces, T_val, n_dims_out)
                  
                   # Compute MSE
                   gpt2_sqerr_val_per_sample = np.sum((gpt2_val_preds - val_targets) ** 2, axis=-1)  # (n_configs, n_traces, T)
                  
                   # Apply mask
                   gpt2_sqerr_val_per_sample_masked = gpt2_sqerr_val_per_sample.copy()
                   gpt2_sqerr_val_per_sample_masked[val_mask] = np.nan
                  
                   # Compute time-series MSE: median across traces first, then across configs for each time step
                   gpt2_trace_medians_val = np.nanmedian(gpt2_sqerr_val_per_sample_masked, axis=1)  # (n_configs, T)
                   gpt2_mse_over_time = np.nanmedian(gpt2_trace_medians_val, axis=0)  # Shape: (T_val,)
                  
                   gpt2_config_medians_val = np.nanmedian(gpt2_trace_medians_val, axis=1)  # (n_configs,)
                   gpt2_final_median_val = np.nanmedian(gpt2_config_medians_val)
                  
                   # Compute event errors
                   gpt2_event_errors = compute_event_errors(gpt2_val_preds, val_targets, val_inputs, val_mask, config, n_dims_in, val_data)
                  
                   batch_haystack_results[hay_len] = {
                       'validation_mse': gpt2_final_median_val,
                       'event_errors': gpt2_event_errors,
                       'mse_over_time': gpt2_mse_over_time
                   }
                  
                   print(f"    Haystack {hay_len}: MSE = {gpt2_final_median_val:.6f}")
              
           except Exception as e:
               print(f"    ❌ Error testing haystack {hay_len}: {e}")
               print(f"    Continuing with next haystack length...")
               batch_haystack_results[hay_len] = None
      
       # Store results for this batch
       batch_results[actual_batch_idx + 1] = {
           'traces_processed': total_traces_processed,  # Cumulative traces across all batches
           'W_opt': W_opt_current,
           'YA_T': YA_T.copy(),
           'AA_T': AA_T.copy(),
           'haystack_results': batch_haystack_results,
           'gpt2_pure_results': gpt2_pure_results if gpt2_pure_computed else None  # Include GPT2 Pure baselines
       }
      
       # Save event errors to trace-based dictionary
       print(f"    💾 Saving event errors for {total_traces_processed:,} traces...")
       cache_dir = "TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k"
       event_errors_file = os.path.join(cache_dir, "event_errors_by_traces_random.npz")
       
       # Load existing event errors if file exists
       if os.path.exists(event_errors_file):
           try:
               existing_data = np.load(event_errors_file, allow_pickle=True)
               event_errors_dict = existing_data['event_errors_dict'].item()
           except:
               event_errors_dict = {}
       else:
           event_errors_dict = {}
       
       # Add current batch event errors to dictionary
       for hay_len in haystack_lengths:
           if batch_haystack_results[hay_len] is not None:
               trace_key = f"{total_traces_processed:06d}"
               if trace_key not in event_errors_dict:
                   event_errors_dict[trace_key] = {}
               
               event_errors_dict[trace_key][f"haystack_{hay_len}"] = {
                   'event_errors': batch_haystack_results[hay_len]['event_errors'],
                   'validation_mse': batch_haystack_results[hay_len]['validation_mse']
               }
       
       # Save updated event errors dictionary
       try:
           np.savez_compressed(event_errors_file, event_errors_dict=event_errors_dict)
           print(f"     Event errors saved for {total_traces_processed:,} traces")
       except Exception as e:
           print(f"     ❌ Error saving event errors: {e}")
       
       # Save latest matrices separately
       print(f"    💾 Saving latest matrices...")
       matrices_file = os.path.join(cache_dir, "latest_matrices_random.npz")
       try:
           np.savez_compressed(
               matrices_file,
               YA_T=YA_T,
               AA_T=AA_T,
               traces_processed=total_traces_processed,
               batch_number=actual_batch_idx + 1,
               timestamp=time.time()
           )
           print(f"     Latest matrices saved")
       except Exception as e:
           print(f"     ❌ Error saving matrices: {e}")
      
       print(f"    📊 Batch {actual_batch_idx + 1} complete! Total traces: {total_traces_processed:,}")
      
       # Individual event error bar graphs removed - only main learning curves kept
      
       
       
       # Update main plots with current batch results
       current_batch_result = {
           actual_batch_idx + 1: batch_results[actual_batch_idx + 1]
       }
       update_main_plots(actual_batch_idx + 1, current_batch_result, haystack_lengths, main_plots, event_types, colors)
      
       # Aggressive memory cleanup after each batch
       import gc
       gc.collect()
       
       # Clear large variables that are no longer needed
       if 'current_batch_activations' in locals():
           del current_batch_activations
       if 'current_batch_targets' in locals():
           del current_batch_targets
       if 'current_batch_masks' in locals():
           del current_batch_masks
       if 'A_current' in locals():
           del A_current
       if 'Y_current' in locals():
           del Y_current
       if 'A_current_unmasked' in locals():
           del A_current_unmasked
       if 'Y_current_unmasked' in locals():
           del Y_current_unmasked
       
       gc.collect()
       print(f"    🧹 Memory cleanup completed for batch {actual_batch_idx + 1}")
      
       # Batch processing complete
  
   # Final W_opt is the last one computed
   W_opt = batch_results[num_batches]['W_opt']
  
   # Store the optimal solution
   pseudoinverse_solutions = {
       'GPT2-Pseudoinverse': W_opt
   }
  
   print("\nOptimal solution computed and stored!")
  
   # ============================================================================
   # FINAL EVALUATION ON HAYSTACK LENGTHS (for comparison with GPT2-Pure)
   # ============================================================================
   haystack_lengths = [1, 2, 5]
   all_results = {}
  
   for hay_len in haystack_lengths:
       print(f"\n" + "="*50)
       print(f"FINAL EVALUATION: Testing haystack length {hay_len}")
       print("="*50)
      
       # Clear any previous data to free memory
       if 'val_data' in locals():
           del val_data
       if 'val_inputs' in locals():
           del val_inputs
       if 'val_targets' in locals():
           del val_targets
       if 'val_mask' in locals():
           del val_mask
       if 'gpt2_val_hidden_acts_optimized' in locals():
           del gpt2_val_hidden_acts_optimized
       if 'gpt2_val_preds' in locals():
           del gpt2_val_preds
       
       # Force garbage collection
       import gc
       gc.collect()
       
       # Load validation data for this haystack length
       val_data_path = f'TFs_do_KF_ICL/src/DataRandomTransformer/val_interleaved_traces_ortho_haar_ident_C_haystack_len_{hay_len}.pkl'
       print(f"Loading validation data from {val_data_path}")
       with open(val_data_path, 'rb') as f:
           val_data = pickle.load(f)
      
       val_inputs = val_data['multi_sys_ys']  # (n_configs,1,n_traces,T,D)
       N_val = np.prod(val_inputs.shape[:-2])
       T_val = val_inputs.shape[-2]
       val_inputs = val_inputs.reshape(N_val, T_val, n_dims_in)
      
       # Get payloads and create targets
       val_payloads = val_inputs[..., -n_dims_out:]
       val_targets = np.zeros_like(val_payloads, dtype=np.float64)
       val_targets[:, :-1, :] = val_payloads[:, 1:, :].astype(np.float64)
      
       # Create validation mask
       val_mask = np.ones((val_inputs.shape[0], val_inputs.shape[1]), dtype=bool)
       val_mask[:, :-1] = (val_inputs[:, 1:, payload_flag_idx] == 0)
       val_mask[:, -1] = True
      
       print(f"Validation data shape: {val_inputs.shape}")
       print(f"Validation mask: masked={val_mask.sum()}  unmasked={(~val_mask).sum()}")
      
       results = {}
      
       # 1. GPT2 with Pseudoinverse (using final W_opt)
       print(f"\n1. GPT2 with Pseudoinverse (haystack_len={hay_len})")
      
       # Recompute validation hidden activations with optimized model
       print("Recomputing GPT2 validation hidden activations with optimized model...")
       gpt2_val_hidden_acts_optimized = []
       gpt2_model.eval()
       with torch.no_grad():
           for i in range(len(val_inputs)):
               x = torch.tensor(val_inputs[i:i+1], dtype=torch.float64).to(device)
               embeds = gpt2_model._read_in(x)
               hidden = gpt2_model._backbone(inputs_embeds=embeds).last_hidden_state
               gpt2_val_hidden_acts_optimized.append(hidden.cpu().numpy().astype(np.float64))
       gpt2_val_hidden_acts_optimized = np.concatenate(gpt2_val_hidden_acts_optimized, axis=0).astype(np.float64)
      
       # Zero out masked hidden states
       gpt2_val_hidden_acts_optimized[val_mask] = 0
      
       # Make predictions using stored W_opt
       gpt2_val_preds = np.einsum('od,ntd->nto', pseudoinverse_solutions['GPT2-Pseudoinverse'], gpt2_val_hidden_acts_optimized)
      
       # Compute MSE
       gpt2_sqerr_val_per_sample = np.sum((gpt2_val_preds - val_targets) ** 2, axis=-1)
       gpt2_sqerr_val_per_sample_reshaped = gpt2_sqerr_val_per_sample.reshape(val_data['multi_sys_ys'].shape[0], -1, T_val)
      
       # Apply mask
       gpt2_val_mask_reshaped = val_mask.reshape(val_data['multi_sys_ys'].shape[0], -1, T_val)
       gpt2_sqerr_val_per_sample_masked = gpt2_sqerr_val_per_sample_reshaped.copy()
       gpt2_sqerr_val_per_sample_masked[gpt2_val_mask_reshaped] = np.nan
      
       # Compute time-series MSE: median across traces, then across configs for each time step
       gpt2_mse_over_time = np.nanmedian(gpt2_sqerr_val_per_sample_masked, axis=(0, 1))  # Shape: (T_val,)
      
       gpt2_trace_medians_val = np.nanmedian(gpt2_sqerr_val_per_sample_masked, axis=2)
       gpt2_config_medians_val = np.nanmedian(gpt2_trace_medians_val, axis=1)
       gpt2_final_median_val = np.nanmedian(gpt2_config_medians_val)
       print(f'GPT2-Pseudoinverse Median MSE (validation): {gpt2_final_median_val:.6f}')
      
       # Compute event errors for GPT2-Pseudoinverse
       gpt2_event_errors = compute_event_errors(gpt2_val_preds, val_targets, val_inputs, val_mask, config, n_dims_in, val_data)
      
       results['GPT2-Pseudoinverse'] = {
           'validation_mse': gpt2_final_median_val,
           'event_errors': gpt2_event_errors,
           'mse_over_time': gpt2_mse_over_time
       }
      
       # 2. GPT2 Pure (using logic from gpt2_pure_evaluation.py)
       # COMMENTED OUT: GPT2 Pure calculations to speed up processing
       # print(f"\n2. GPT2 Pure (Original) (haystack_len={hay_len})")
       # 
       # # Load the same trained GPT2 model from checkpoint for pure evaluation
       # gpt2_checkpoint_path = 'TFs_do_KF_ICL/src/DataRandomTransformer/step%3D99000.ckpt'
       # print(f"Loading GPT2 Pure model from checkpoint: {gpt2_checkpoint_path}")
       # 
       # try:
       #     gpt2_pure_model = GPT2.load_from_checkpoint(
       #         gpt2_checkpoint_path,
       #         n_dims_in=n_dims_in,
       #         n_positions=n_positions,
       #         n_embd=n_embd,
       #         n_layer=n_layer,
       #         n_head=n_head,
       #         n_dims_out=n_dims_out
       #     )
       #     print("Successfully loaded GPT2 Pure checkpoint")
       # except Exception as e:
       #     print(f"Failed to load checkpoint: {e}")
       #     print("Creating new GPT2 model as fallback")
       #     gpt2_pure_model = GPT2(
       #         n_dims_in=n_dims_in,
       #         n_positions=n_positions,
       #         n_embd=n_embd,
       #         n_layer=n_layer,
       #         n_head=n_head,
       #         n_dims_out=n_dims_out
       #     )
       # 
       # gpt2_pure_model = gpt2_pure_model.to(device)
       # gpt2_pure_model = gpt2_pure_model.double()
       # 
       # # Evaluate using model's forward pass (consistent with gpt2_pure_evaluation.py)
       # gpt2_pure_val_preds = []
       # gpt2_pure_model.eval()
       # with torch.no_grad():
       #     for i in range(len(val_inputs)):
       #         x = torch.tensor(val_inputs[i:i+1], dtype=torch.float64).to(device)
       #         # Get embeddings
       #         embeds = gpt2_pure_model._read_in(x)
       #         # Get hidden states from backbone
       #         hidden = gpt2_pure_model._backbone(inputs_embeds=embeds).last_hidden_state
       #         # Get predictions from read_out layer
       #         pred = gpt2_pure_model._read_out(hidden)  # Shape: (1, T, 5)
       #         gpt2_pure_val_preds.append(pred.cpu().numpy().astype(np.float64))
       # gpt2_pure_val_preds = np.concatenate(gpt2_pure_val_preds, axis=0).astype(np.float64)  # Shape: (N_val, T_val, 5)
       # 
       # # Compute MSE
       # gpt2_pure_sqerr_val_per_sample = np.sum((gpt2_pure_val_preds - val_targets) ** 2, axis=-1)
       # gpt2_pure_sqerr_val_per_sample_reshaped = gpt2_pure_sqerr_val_per_sample.reshape(val_data['multi_sys_ys'].shape[0], -1, T_val)
       # 
       # # Apply mask
       # gpt2_pure_val_mask_reshaped = val_mask.reshape(val_data['multi_sys_ys'].shape[0], -1, T_val)
       # gpt2_pure_sqerr_val_per_sample_masked = gpt2_pure_sqerr_val_per_sample_reshaped.copy()
       # gpt2_pure_sqerr_val_per_sample_masked[gpt2_pure_val_mask_reshaped] = np.nan
       # 
       # # Compute time-series MSE: median across traces, then across configs for each time step
       # gpt2_pure_mse_over_time = np.nanmedian(gpt2_pure_sqerr_val_per_sample_masked, axis=(0, 1))  # Shape: (T_val,)
       # 
       # gpt2_pure_trace_medians_val = np.nanmedian(gpt2_pure_sqerr_val_per_sample_masked, axis=2)
       # gpt2_pure_config_medians_val = np.nanmedian(gpt2_pure_trace_medians_val, axis=1)
       # gpt2_pure_final_median_val = np.nanmedian(gpt2_pure_config_medians_val)
       # print(f'GPT2-Pure Median MSE (validation): {gpt2_pure_final_median_val:.6f}')
       # 
       # # Compute event errors for GPT2-Pure
       # gpt2_pure_event_errors = compute_event_errors(gpt2_pure_val_preds, val_targets, val_inputs, val_mask, config, n_dims_in, val_data)
       # 
       # results['GPT2-Pure'] = {
       #     'validation_mse': gpt2_pure_final_median_val,
       #     'event_errors': gpt2_pure_event_errors,
       #     'mse_over_time': gpt2_pure_mse_over_time
       # }
       
       # Skip GPT2 Pure computation for faster processing
       print(f"\n⏭️  Skipping GPT2 Pure computation for faster processing")
      
       # Store results for this haystack length
       all_results[hay_len] = results
      
       # Print summary for this haystack length
       print(f"\n" + "="*50)
       print(f"SUMMARY for haystack_len={hay_len}")
       print("="*50)
       for model_name, result in results.items():
           print(f"{model_name}: Validation MSE: {result['validation_mse']:.6f}")
       print()
  
   # Individual learning curves removed - only main learning curves kept
  
   # Save the final results to the 20k batches directory
   results_file = f'TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k/unified_model_comparison_results_{total_traces:,}_traces_random.pkl'
   with open(results_file, 'wb') as f:
       pickle.dump({
           'all_results': all_results,
           'batch_results': batch_results,
           'pseudoinverse_solutions': pseudoinverse_solutions,
           'YA_T_final': YA_T,
           'AA_T_final': AA_T,
           'total_traces': total_traces
       }, f)
  
   print(f"\nFinal results saved to {results_file}")
   print(f"Total traces processed: {total_traces}")
   print(f"Final YA_T shape: {YA_T.shape}")
   print(f"Final AA_T shape: {AA_T.shape}")
  
   # Print learning curve summary
   print(f"\n" + "="*60)
   print("LEARNING CURVE SUMMARY")
   print("="*60)
   for hay_len in haystack_lengths:
       print(f"\nHaystack length {hay_len}:")
       for batch_num in range(1, num_batches + 1):
           if batch_num in batch_results and batch_results[batch_num]['haystack_results'] and hay_len in batch_results[batch_num]['haystack_results']:
               if batch_results[batch_num]['haystack_results'][hay_len] is not None:
                   traces = batch_num * 20000  # Each batch has 20k traces
                   mse = batch_results[batch_num]['haystack_results'][hay_len]['validation_mse']
                   print(f"  {traces:6d} traces: MSE = {mse:.6f}")
   
   # Save final plots
   print(f"\n" + "="*60)
   print("SAVING FINAL PLOTS")
   print("="*60)
   for hay_len in haystack_lengths:
       if hay_len in main_plots:
           plot_data = main_plots[hay_len]
           plt.figure(plot_data['fig'])
           final_plot_path = f"TFs_do_KF_ICL/src/DataRandomTransformer/batched_20k/main_learning_curves_haystack_{hay_len}_random.png"
           plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
           print(f"Saved final plot for haystack {hay_len}: {final_plot_path}")
   
   # Close all plots
   for hay_len in haystack_lengths:
       if hay_len in main_plots:
           plot_data = main_plots[hay_len]
           plt.close(plot_data['fig'])
   plt.close('all')
   print("All plots closed and saved.")


def compute_event_errors(preds, targets, inputs, mask, config, n_dims_in, val_data):
   """Compute per-event errors using corrected system indexing logic"""
   event_types = [f"{k}_after_initial" for k in [1,2,3,7,8]] + [f"{k}_after_final" for k in [1,2,3,7,8]]
   n_events = len(event_types)
   
   # Only handle 4D data structure: (n_configs, n_traces, T, D)
   n_configs, n_traces, T = preds.shape[:3]
   event_errors = np.full((n_configs, n_traces, n_events), np.nan)
   
   for config_i in range(n_configs):
       for trace_i in range(n_traces):
           # Find the first system that starts this trace (first even-indexed nonzero element in positions 0-49)
           first_system_open_idx = None
           for pos in range(T):
               for sys_idx in range(50):  # Check positions 0-49 ONLY (exclude position 50)
                   if sys_idx % 2 == 0 and inputs[config_i, trace_i, pos, sys_idx] != 0:  # Even index = open parenthesis
                       first_system_open_idx = sys_idx
                       first_system_open_pos = pos
                       break
               if first_system_open_idx is not None:
                   break
          
           if first_system_open_idx is None:
               continue
          
           # Find all occurrences of this same open parenthesis in the trace
           same_open_events = np.where(inputs[config_i, trace_i, :, first_system_open_idx] != 0)[0]
           if len(same_open_events) < 2:
               continue
          
           p_init = same_open_events[0]    # First occurrence of this open parenthesis
           p_final = same_open_events[-1]  # Last occurrence of this open parenthesis
          
           # After initial: k positions after the first open parenthesis
           for k_idx, k in enumerate([1, 2, 3, 7, 8]):
               idx = p_init + k  # k positions AFTER the open parenthesis
               if idx < T and not mask[config_i, trace_i, idx]:
                   pred = preds[config_i, trace_i, idx]
                   target = targets[config_i, trace_i, idx]
                   sqerr = np.sum((pred - target) ** 2)
                   event_errors[config_i, trace_i, k_idx] = sqerr
          
           # After final: k positions after the last open parenthesis
           for k_idx, k in enumerate([1, 2, 3, 7, 8]):
               idx = p_final + k  # k positions AFTER the open parenthesis
               if idx < T and not mask[config_i, trace_i, idx]:
                   pred = preds[config_i, trace_i, idx]
                   target = targets[config_i, trace_i, idx]
                   sqerr = np.sum((pred - target) ** 2)
                   event_errors[config_i, trace_i, k_idx + 5] = sqerr
  
   # medians over traces first, then configs
   median_over_traces = np.nanmedian(event_errors, axis=1)  # (n_configs,n_events)
   final_median_events = np.nanmedian(median_over_traces, axis=0)  # (n_events,)
  
   return final_median_events


if __name__ == "__main__":
   main()


