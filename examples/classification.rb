#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Run an ONNX classification model
#
# With tokenizer (text input):
#   gem install onnx-ruby tokenizers
#   ruby classification.rb intent_model.onnx bert-base-uncased
#
# With raw features:
#   ruby classification.rb classifier.onnx

require "onnx_ruby"

model_path = ARGV[0] || "classifier.onnx"
tokenizer_name = ARGV[1]
abort "Usage: ruby classification.rb <model.onnx> [tokenizer_name]" unless File.exist?(model_path)

labels = %w[greeting farewell question command]

if tokenizer_name
  classifier = OnnxRuby::Classifier.new(model_path,
                                        tokenizer: tokenizer_name,
                                        labels: labels)
  result = classifier.predict("Hello there!")
  puts "Prediction: #{result[:label]} (score: #{result[:score].round(4)})"
else
  classifier = OnnxRuby::Classifier.new(model_path, labels: labels)

  # Use raw feature vector
  features = Array.new(8) { rand(-1.0..1.0) }
  result = classifier.predict(features)
  puts "Prediction: #{result[:label]} (score: #{result[:score].round(4)})"
  puts "All scores: #{result[:scores].map { |s| s.round(4) }}"
end
