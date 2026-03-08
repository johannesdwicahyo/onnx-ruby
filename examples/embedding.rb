#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Run an ONNX embedding model
#
# With tokenizer (text input):
#   gem install onnx-ruby tokenizers
#   ruby embedding.rb all-MiniLM-L6-v2.onnx sentence-transformers/all-MiniLM-L6-v2
#
# With raw token IDs:
#   ruby embedding.rb all-MiniLM-L6-v2.onnx

require "onnx_ruby"

model_path = ARGV[0] || "all-MiniLM-L6-v2.onnx"
tokenizer_name = ARGV[1]
abort "Usage: ruby embedding.rb <model.onnx> [tokenizer_name]" unless File.exist?(model_path)

if tokenizer_name
  embedder = OnnxRuby::Embedder.new(model_path, tokenizer: tokenizer_name)
  embedding = embedder.embed("Hello world")
  puts "Embedding (#{embedding.length} dims): #{embedding.take(5).map { |v| v.round(4) }}..."

  batch = embedder.embed_batch(["Hello", "World", "Ruby is great"])
  puts "Batch: #{batch.length} embeddings of #{batch.first.length} dims"
else
  embedder = OnnxRuby::Embedder.new(model_path)

  # Use pre-tokenized input
  result = embedder.embed({
    "input_ids" => [101, 2023, 2003, 1037, 3231, 102],
    "attention_mask" => [1, 1, 1, 1, 1, 1]
  })
  puts "Embedding (#{result.length} dims): #{result.take(5).map { |v| v.round(4) }}..."
end
