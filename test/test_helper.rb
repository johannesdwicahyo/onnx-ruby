# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "onnx_ruby"
require "minitest/autorun"

module TestHelper
  MODELS_DIR = File.expand_path("models", __dir__)

  def simple_model_path
    File.join(MODELS_DIR, "simple.onnx")
  end

  def model_exists?(name)
    File.exist?(File.join(MODELS_DIR, name))
  end
end
