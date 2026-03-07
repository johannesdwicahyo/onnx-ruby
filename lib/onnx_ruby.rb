# frozen_string_literal: true

require_relative "onnx_ruby/version"
require_relative "onnx_ruby/onnx_ruby_ext"
require_relative "onnx_ruby/tensor"
require_relative "onnx_ruby/session"

module OnnxRuby
  class Error < StandardError; end
  class ModelError < Error; end
  class InferenceError < Error; end
  class TensorError < Error; end
end
