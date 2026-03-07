#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Run an ONNX classification model
#
# Prerequisites:
#   - Export a classification model to ONNX format
#   - gem install onnx-ruby

require "onnx_ruby"

model_path = ARGV[0] || "classifier.onnx"
abort "Usage: ruby classification.rb <model.onnx>" unless File.exist?(model_path)

session = OnnxRuby::Session.new(model_path)

puts "Model inputs:"
session.inputs.each { |i| puts "  #{i[:name]}: #{i[:type]} #{i[:shape]}" }

puts "\nModel outputs:"
session.outputs.each { |o| puts "  #{o[:name]}: #{o[:type]} #{o[:shape]}" }
