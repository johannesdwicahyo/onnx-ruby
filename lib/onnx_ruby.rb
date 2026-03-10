# frozen_string_literal: true

require_relative "onnx_ruby/version"

module OnnxRuby
  class Error < StandardError; end
  class ModelError < Error; end
  class InferenceError < Error; end
  class TensorError < Error; end
end

require_relative "onnx_ruby/onnx_ruby_ext"
require_relative "onnx_ruby/tensor"
require_relative "onnx_ruby/session"
require_relative "onnx_ruby/tokenizer_support"
require_relative "onnx_ruby/embedder"
require_relative "onnx_ruby/classifier"
require_relative "onnx_ruby/reranker"
require_relative "onnx_ruby/hub"
require_relative "onnx_ruby/configuration"
require_relative "onnx_ruby/lazy_session"
require_relative "onnx_ruby/session_pool"
require_relative "onnx_ruby/model"

module OnnxRuby
  class << self
    def configuration
      @configuration ||= Configuration.new
    end

    def configure
      yield configuration
    end
  end

  def self.optimize(input_path, output_path, level: :all)
    input_path = File.expand_path(input_path)
    raise ModelError, "model file not found: #{input_path}" unless File.exist?(input_path)

    Ext.optimize_model(input_path, output_path, level.to_s)
  end

  def self.available_providers
    Ext.available_providers
  end
end
