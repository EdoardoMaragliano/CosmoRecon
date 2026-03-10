   ┌─────────────────────────────┐
   │ Observed .npy files (x)     │
   │ True .npy files (y)         │
   │ Mask .npy files (mask)      │
   └─────────────┬───────────────┘
                 │
                 v
   ┌─────────────────────────────┐
   │ tf.data.Dataset             │
   │ - from_tensor_slices        │
   │ - map(parse_fn)             │
   │ - batch(batch_size)         │
   │ - prefetch(AUTOTUNE)        │
   └─────────────┬───────────────┘
                 │
                 v
   ┌─────────────────────────────┐
   │ Input to U-Net:             │
   │ x_sample = [observed, mask] │
   └─────────────┬───────────────┘
                 │
                 v
   ┌─────────────────────────────┐
   │ Forward Pass through U-Net  │
   │ Output: y_pred              │
   └─────────────┬───────────────┘
                 │
                 v
   ┌─────────────────────────────┐
   │ Compute Masked Loss         │
   │ loss = mean((y_true-y_pred)^2 * (1-mask)) │
   └─────────────┬───────────────┘
                 │
                 v
   ┌─────────────────────────────┐
   │ GradientTape                │
   │ - compute grads(loss)       │
   │ - optimizer.apply_gradients │
   └─────────────┬───────────────┘
                 │
                 v
   ┌─────────────────────────────┐
   │ Update model weights        │
   └─────────────┬───────────────┘
                 │
                 v
   ┌─────────────────────────────┐
   │ Repeat for next batch       │
   │ Callbacks: ModelCheckpoint │
   │ TensorBoard logging         │
   └─────────────────────────────┘
