# Experiment 03: efficiency profile

The profile uses isolated interpreters for static graph FLOPs and GPU runtime.
FLOPs are TensorFlow graph floating-point operations at batch size 1; MACs are
reported as the estimate FLOPs / 2. Runtime uses 100 warm-up iterations and
1000 synchronized timed iterations on the recorded GPU environment.

No training or data loading occurs in the profiler. The profile is bound to an
existing Stage-2 checkpoint by SHA-256; the checkpoint itself is excluded from
this Git package.
