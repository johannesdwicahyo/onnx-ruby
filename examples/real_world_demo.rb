#!/usr/bin/env ruby
# frozen_string_literal: true

# Real-world proof of concept: Semantic search with all-MiniLM-L6-v2
#
# This demo:
# 1. Loads a real HuggingFace sentence-transformers model (86MB)
# 2. Tokenizes English text with the tokenizers gem
# 3. Generates 384-dimensional embeddings
# 4. Performs cosine similarity search
#
# Prerequisites:
#   gem install tokenizers

require_relative "../lib/onnx_ruby"
require "tokenizers"
require "benchmark"

MODEL_PATH = File.join(__dir__, "models", "all-MiniLM-L6-v2.onnx")
TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"

unless File.exist?(MODEL_PATH)
  abort "Model not found. Download it first:\n" \
        "  curl -fSL https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx " \
        "-o examples/models/all-MiniLM-L6-v2.onnx"
end

# --- Setup ---

puts "Loading tokenizer..."
tokenizer = Tokenizers::Tokenizer.from_pretrained(TOKENIZER_ID)

puts "Loading ONNX model..."
embedder = OnnxRuby::Embedder.new(MODEL_PATH, tokenizer: tokenizer, normalize: true)

puts "Model inputs:  #{embedder.session.inputs.map { |i| "#{i[:name]} #{i[:type]} #{i[:shape]}" }.join(", ")}"
puts "Model outputs: #{embedder.session.outputs.map { |o| "#{o[:name]} #{o[:type]} #{o[:shape]}" }.join(", ")}"
puts

# --- Document corpus ---

documents = [
  "Ruby is a dynamic, open source programming language with a focus on simplicity and productivity.",
  "Python is widely used in data science and machine learning applications.",
  "ONNX Runtime is a cross-platform inference engine for machine learning models.",
  "Rails is a web application framework written in Ruby.",
  "Vector databases store and search high-dimensional embeddings efficiently.",
  "The weather in Tokyo is warm and humid during summer.",
  "PostgreSQL is a powerful open source relational database system.",
  "Transformers have revolutionized natural language processing since 2017.",
  "Docker containers package applications with their dependencies for consistent deployment.",
  "The quick brown fox jumps over the lazy dog.",
  "Semantic search understands the meaning of queries, not just keywords.",
  "GPUs accelerate matrix operations used in deep learning training.",
]

# --- Generate embeddings ---

puts "Generating embeddings for #{documents.length} documents..."
doc_embeddings = nil
time = Benchmark.realtime do
  doc_embeddings = embedder.embed_batch(documents)
end
puts "  Done in #{(time * 1000).round(1)}ms (#{(time / documents.length * 1000).round(1)}ms per document)"
puts "  Embedding dimensions: #{doc_embeddings.first.length}"
puts

# --- Cosine similarity search ---

def cosine_similarity(a, b)
  dot = a.zip(b).sum { |x, y| x * y }
  norm_a = Math.sqrt(a.sum { |x| x * x })
  norm_b = Math.sqrt(b.sum { |x| x * x })
  dot / (norm_a * norm_b)
end

queries = [
  "How do I build a web app with Ruby?",
  "What is the best way to run ML models locally?",
  "Tell me about database systems",
  "Natural language understanding",
]

queries.each do |query|
  puts "Query: \"#{query}\""

  query_embedding = embedder.embed(query)

  results = documents.each_with_index.map do |doc, i|
    { document: doc, score: cosine_similarity(query_embedding, doc_embeddings[i]), index: i }
  end.sort_by { |r| -r[:score] }

  results.first(3).each_with_index do |r, rank|
    puts "  #{rank + 1}. [#{r[:score].round(4)}] #{r[:document]}"
  end
  puts
end

# --- Classifier demo ---

puts "=== Classifier Demo ==="
classifier = OnnxRuby::Classifier.new(
  File.join(__dir__, "..", "test", "models", "classifier.onnx"),
  labels: %w[greeting farewell question command]
)

# Use embeddings as features (truncated to 8 dims to match test model)
test_sentences = ["Hello there!", "What is Ruby?", "Goodbye!", "Run the tests"]
test_sentences.each do |sentence|
  emb = embedder.embed(sentence)
  pred = classifier.predict(emb.first(8))  # our test classifier expects 8 features
  puts "  \"#{sentence}\" -> #{pred[:label]} (#{(pred[:score] * 100).round(1)}%)"
end
puts

# --- Session Pool demo ---

puts "=== Concurrent Inference with SessionPool ==="
pool = OnnxRuby::SessionPool.new(MODEL_PATH, size: 3)

sentences = [
  "Machine learning is transforming software",
  "Ruby gems make code reusable",
  "ONNX is an open format for ML models",
  "Concurrent processing improves throughput",
  "Embeddings capture semantic meaning",
  "Thread safety matters in production",
]

results = []
mutex = Mutex.new
time = Benchmark.realtime do
  threads = sentences.map do |sentence|
    Thread.new do
      encoding = tokenizer.encode(sentence)
      ids = encoding.ids
      mask = encoding.attention_mask

      r = pool.run({
        "input_ids" => [ids],
        "attention_mask" => [mask],
        "token_type_ids" => [Array.new(ids.length, 0)]
      })

      embedding = r.values.first[0]
      mutex.synchronize { results << { sentence: sentence, dims: embedding.length } }
    end
  end
  threads.each(&:join)
end

puts "  #{results.length} sentences embedded concurrently in #{(time * 1000).round(1)}ms"
puts "  Pool size: #{pool.size} sessions created (max: 3)"
results.each { |r| puts "    #{r[:sentence]} -> #{r[:dims]}d" }
puts

# --- Lazy loading demo ---

puts "=== Lazy Loading Demo ==="
lazy = OnnxRuby::LazySession.new(MODEL_PATH)
puts "  Before first call: loaded=#{lazy.loaded?}"
lazy.run({
  "input_ids" => [[101, 2023, 2003, 1037, 3231, 102]],
  "attention_mask" => [[1, 1, 1, 1, 1, 1]],
  "token_type_ids" => [[0, 0, 0, 0, 0, 0]]
})
puts "  After first call:  loaded=#{lazy.loaded?}"
puts

puts "=== All real-world demos completed successfully! ==="
