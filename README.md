# onnx-ruby

High-performance [ONNX Runtime](https://onnxruntime.ai/) bindings for Ruby. Run ONNX models locally for embeddings, classification, reranking, and any other ML inference — without Python or API calls.

Built with [Rice](https://github.com/ruby-rice/rice) (C++ to Ruby bindings) wrapping the ONNX Runtime C++ API directly.

## Features

- **Fast inference** — native C++ bindings, not FFI
- **Auto-download** — ONNX Runtime is downloaded automatically during gem install
- **Multiple providers** — CPU, CoreML (macOS), CUDA, TensorRT
- **High-level wrappers** — `Embedder`, `Classifier`, `Reranker` for common ML tasks
- **Thread-safe** — `SessionPool` for concurrent inference in multi-threaded apps
- **Lazy loading** — `LazySession` loads models on first use
- **Rails-ready** — `OnnxRuby::Model` mixin, global configuration, connection pooling
- **Model hub** — download models from HuggingFace with local caching

## Installation

```ruby
# Gemfile
gem "onnx-ruby"
```

```sh
bundle install
```

ONNX Runtime (v1.24.3) is automatically downloaded during native extension compilation.

To use a custom ONNX Runtime installation:

```sh
ONNX_RUNTIME_DIR=/path/to/onnxruntime bundle install
```

## Quick Start

### Basic Inference

```ruby
require "onnx_ruby"

# Load a model
session = OnnxRuby::Session.new("model.onnx")

# Inspect model
session.inputs   # => [{ name: "input", type: :float32, shape: [-1, 4] }]
session.outputs  # => [{ name: "output", type: :float32, shape: [-1, 3] }]

# Run inference
result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0]] })
result["output"]  # => [[0.123, -0.456, 0.789]]

# Batch inference
result = session.run({ "input" => [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]] })
result["output"]  # => [[...], [...]]
```

### Embeddings

```ruby
require "onnx_ruby"
require "tokenizers"

# With a HuggingFace tokenizer
tokenizer = Tokenizers::Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = OnnxRuby::Embedder.new("all-MiniLM-L6-v2.onnx",
  tokenizer: tokenizer,
  normalize: true
)

# Single embedding
embedding = embedder.embed("Hello world")  # => [0.0123, -0.0456, ...] (384 dims)

# Batch embedding
embeddings = embedder.embed_batch(["Hello", "World"])  # => [[...], [...]]

# Without tokenizer (pre-tokenized input)
embedder = OnnxRuby::Embedder.new("model.onnx")
embedding = embedder.embed({
  "input_ids" => [101, 2023, 2003, 102],
  "attention_mask" => [1, 1, 1, 1]
})
```

### Classification

```ruby
classifier = OnnxRuby::Classifier.new("classifier.onnx",
  labels: ["greeting", "farewell", "question", "command"]
)

# With feature vectors
result = classifier.predict([0.1, 0.2, 0.3, 0.4])
# => { label: "greeting", score: 0.95, scores: [0.95, 0.02, 0.02, 0.01] }

# Batch
results = classifier.predict_batch([features1, features2])

# With tokenizer for text input
classifier = OnnxRuby::Classifier.new("bert-classifier.onnx",
  tokenizer: "bert-base-uncased",
  labels: ["positive", "negative"]
)
classifier.predict("This is great!")
```

### Reranking

```ruby
reranker = OnnxRuby::Reranker.new("reranker.onnx", tokenizer: tokenizer)

# Rerank documents by relevance to a query
results = reranker.rerank("What is Ruby?", [
  "Ruby is a programming language",
  "The weather is nice today",
  "Rails is built with Ruby"
])
# => [
#   { document: "Ruby is a programming language", score: 0.98, index: 0 },
#   { document: "Rails is built with Ruby", score: 0.85, index: 2 },
#   { document: "The weather is nice today", score: 0.01, index: 1 }
# ]

# Raw scoring with pre-tokenized inputs
scores = reranker.score(
  input_ids: [[101, 2023, 102], [101, 7592, 102]],
  attention_mask: [[1, 1, 1], [1, 1, 1]]
)
```

## Session Options

```ruby
session = OnnxRuby::Session.new("model.onnx",
  providers: [:coreml, :cpu],       # execution providers (fallback order)
  optimization_level: :all,          # :none, :basic, :extended, :all
  intra_threads: 4,                  # threads within an operator
  inter_threads: 2,                  # threads between operators
  execution_mode: :parallel,         # :sequential or :parallel
  memory_pattern: true,              # pre-allocate memory
  cpu_mem_arena: true,               # use memory arena
  log_level: :warning                # :verbose, :info, :warning, :error, :fatal
)
```

### Execution Providers

```ruby
# List available providers
OnnxRuby.available_providers
# => ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# CoreML (macOS — uses Apple Neural Engine)
session = OnnxRuby::Session.new("model.onnx", providers: [:coreml])

# CUDA (NVIDIA GPU — requires CUDA build of ONNX Runtime)
session = OnnxRuby::Session.new("model.onnx", providers: [:cuda, :cpu])
```

## Model Optimization

```ruby
# Optimize and save a model
OnnxRuby.optimize("model.onnx", "model_optimized.onnx", level: :all)

# Use the optimized model
session = OnnxRuby::Session.new("model_optimized.onnx")
```

## Tensors

```ruby
# Create typed tensors
tensor = OnnxRuby::Tensor.new([1, 2, 3, 4], shape: [2, 2], dtype: :int64)
tensor.to_a     # => [[1, 2], [3, 4]]
tensor.shape    # => [2, 2]
tensor.dtype    # => :int64

# Convenience constructors
OnnxRuby::Tensor.float([0.1, 0.2, 0.3], shape: [1, 3])
OnnxRuby::Tensor.int64([1, 2, 3], shape: [3])
OnnxRuby::Tensor.double([1.0, 2.0], shape: [2])
OnnxRuby::Tensor.int32([1, 2], shape: [2])

# Use tensors as session input
tensor = OnnxRuby::Tensor.float([1.0, 2.0, 3.0, 4.0], shape: [1, 4])
session.run({ "input" => tensor })
```

Supported dtypes: `float32`, `float64`, `int32`, `int64`, `bool`, `string`

## Thread Safety

### Session Pool

```ruby
# Create a pool of sessions for concurrent inference
pool = OnnxRuby::SessionPool.new("model.onnx", size: 5, timeout: 10)

# Auto checkout/checkin
result = pool.run({ "input" => data })

# Or manual block form
pool.with_session do |session|
  session.run({ "input" => data })
end

# Pool stats
pool.size       # => number of created sessions
pool.available  # => number of idle sessions
```

### Lazy Loading

```ruby
# Model loads on first use, thread-safe
lazy = OnnxRuby::LazySession.new("model.onnx")
lazy.loaded?  # => false
lazy.run(inputs)
lazy.loaded?  # => true
```

## Rails Integration

### Configuration

```ruby
# config/initializers/onnx_ruby.rb
OnnxRuby.configure do |c|
  c.models_path = "app/models/onnx"
  c.default_providers = [:coreml, :cpu]
  c.default_log_level = :warning
  c.pool_size = 5
  c.pool_timeout = 5
end
```

### ActiveModel Mixin

```ruby
class Document < ApplicationRecord
  include OnnxRuby::Model

  onnx_model "embeddings.onnx"
  onnx_input ->(doc) {
    # tokenize doc.content and return input hash
    { "input_ids" => ids, "attention_mask" => mask }
  }
  onnx_output "embeddings"

  def generate_embedding
    self.embedding = onnx_predict.first
  end
end

doc = Document.find(1)
doc.generate_embedding  # runs ONNX inference
```

The model is loaded lazily on first inference and shared across all instances.

## Model Hub

```ruby
# Download from HuggingFace
path = OnnxRuby::Hub.download("sentence-transformers/all-MiniLM-L6-v2",
  filename: "model.onnx"
)
session = OnnxRuby::Session.new(path)

# Cache management
OnnxRuby::Hub.cached_models  # => ["/home/user/.cache/onnx_ruby/models/..."]
OnnxRuby::Hub.clear_cache
```

## Requirements

- Ruby >= 3.1
- C++ compiler with C++17 support
- ONNX Runtime (auto-downloaded during install)

### Optional

- [tokenizers](https://github.com/ankane/tokenizers-ruby) gem — for text tokenization in Embedder/Classifier/Reranker

## Development

```sh
git clone https://github.com/johannesdwicahyo/onnx-ruby.git
cd onnx-ruby
bundle install
bundle exec rake compile
python3 script/create_test_models.py  # requires torch, onnx, onnxscript
bundle exec rake test
```

## License

MIT License. See [LICENSE](LICENSE).
