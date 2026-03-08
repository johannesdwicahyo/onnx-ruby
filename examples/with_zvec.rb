#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Full RAG pipeline with onnx-ruby + zvec-ruby
#
# Prerequisites:
#   gem install onnx-ruby zvec-ruby tokenizers
#   Download or export an ONNX embedding model

require "onnx_ruby"

begin
  require "zvec"
rescue LoadError
  abort "This example requires zvec-ruby: gem install zvec-ruby"
end

MODEL_PATH = ARGV[0] || "all-MiniLM-L6-v2.onnx"
TOKENIZER = ARGV[1] || "sentence-transformers/all-MiniLM-L6-v2"

unless File.exist?(MODEL_PATH)
  abort "Usage: ruby with_zvec.rb <embedding_model.onnx> [tokenizer_name]"
end

# 1. Create embedder
embedder = OnnxRuby::Embedder.new(MODEL_PATH, tokenizer: TOKENIZER)

# 2. Create vector store
dim = embedder.session.outputs.first[:shape].last
store = Zvec::Store.new(dimensions: dim)

# 3. Index some documents
documents = [
  "Ruby is a dynamic programming language",
  "ONNX Runtime provides high-performance inference",
  "Vector databases enable semantic search",
  "Machine learning models can run locally",
  "Rails is a web application framework"
]

embeddings = embedder.embed_batch(documents)
documents.each_with_index do |doc, i|
  store.add(embeddings[i], metadata: { text: doc, id: i })
end

# 4. Query
query = "How to run ML models?"
query_embedding = embedder.embed(query)
results = store.search(query_embedding, k: 3)

puts "Query: #{query}\n\n"
results.each_with_index do |result, i|
  puts "#{i + 1}. (score: #{result[:score].round(4)}) #{result[:metadata][:text]}"
end
