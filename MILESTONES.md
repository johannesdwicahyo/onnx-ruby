# onnx-ruby — Milestones

> **Source of truth:** https://github.com/johannesdwicahyo/onnx-ruby/milestones
> **Last synced:** 2026-04-14

This file mirrors the GitHub milestones for this repo. Edit the milestone or issues on GitHub and re-sync, do not hand-edit.

## v1.0.0 — Production Ready (**open**)

_Production hardening, full documentation, stable API_

- [ ] #31 YARD documentation for all public APIs
- [ ] #32 Add CHANGELOG.md
- [ ] #33 Thread safety audit and documentation
- [ ] #34 Memory leak audit with Valgrind/LeakSanitizer
- [ ] #35 Rails integration guide and example app
- [ ] #36 Benchmark suite vs Python onnxruntime
- [ ] #37 Stable API — freeze public interface
- [ ] #38 Wrap ORT exceptions in typed Ruby exceptions

## v0.5.0 — Advanced Features (**open**)

_Quantization, profiling, IO binding, custom operators_

- [ ] #24 Add model profiling and benchmarking
- [ ] #25 Add quantized model support (int8/uint8 inference)
- [ ] #26 Add Numo::NArray integration for zero-copy tensors
- [ ] #27 Add IO binding for GPU memory
- [ ] #28 Add custom operator registration
- [ ] #29 Add Classifier multi-label and top-k support
- [ ] #30 Add Embedder similarity helper methods

## v0.4.0 — Performance & Platform (**open**)

_Performance optimizations, Windows support, precompiled gems_

- [ ] #17 Ship precompiled native gems for macOS and Linux
- [ ] #18 Add Windows platform support
- [ ] #19 Reduce Ruby→C++ data copy overhead
- [ ] #20 Add Hub download retry, timeout, and checksum verification
- [ ] #21 Add CUDA/TensorRT provider fallback to CPU
- [ ] #22 Validate providers are available before session creation
- [ ] #23 Cap Rice version constraint in gemspec

## v0.3.0 — Test Coverage & Edge Cases (**open**)

_Comprehensive test coverage, edge case handling, CI/CD pipeline_

- [ ] #9 Add tests for string and bool tensor types
- [ ] #10 Add tests for error conditions and invalid inputs
- [ ] #11 Add tests for Embedder with 3D output (mean pooling)
- [ ] #12 Add tests for Hub download failures and edge cases
- [ ] #13 Add tests for LazySession concurrent loading race
- [ ] #14 Set up GitHub Actions CI pipeline
- [ ] #15 Handle empty tensor edge case
- [ ] #16 Add Configuration validation

## v0.2.0 — Stability & Safety (**closed**)

_Fix critical bugs, harden error handling, improve type safety in C++ extension_

- [x] #1 Fix bool tensor undefined behavior in C++ extension
- [x] #2 Add uint8/uint16/int8/int16/float16 output tensor support
- [x] #3 Validate input shape matches data size before C++ call
- [x] #4 Handle jagged/uneven arrays in shape inference
- [x] #5 Protect against shape dimension overflow in C++
- [x] #6 Fix SessionPool condition variable: signal vs broadcast
- [x] #7 Fix SessionPool created counter leak on session creation failure
- [x] #8 Add softmax numerical stability for extreme logits
