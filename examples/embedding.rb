#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Run an ONNX embedding model
#
# Prerequisites:
#   - Export an embedding model to ONNX format (e.g., all-MiniLM-L6-v2)
#   - gem install onnx-ruby

require "onnx_ruby"

model_path = ARGV[0] || "all-MiniLM-L6-v2.onnx"
abort "Usage: ruby embedding.rb <model.onnx>" unless File.exist?(model_path)

session = OnnxRuby::Session.new(model_path)

puts "Model inputs:"
session.inputs.each { |i| puts "  #{i[:name]}: #{i[:type]} #{i[:shape]}" }

puts "\nModel outputs:"
session.outputs.each { |o| puts "  #{o[:name]}: #{o[:type]} #{o[:shape]}" }

# Example: run with token IDs (you'd normally get these from a tokenizer)
# input_ids = [[101, 2023, 2003, 1037, 3231, 102]]
# attention_mask = [[1, 1, 1, 1, 1, 1]]
#
# result = session.run({
#   "input_ids" => input_ids,
#   "attention_mask" => attention_mask
# })
# puts result["embeddings"][0].take(5)
