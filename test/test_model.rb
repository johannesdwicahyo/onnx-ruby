# frozen_string_literal: true

require "test_helper"

class TestModel < Minitest::Test
  include TestHelper

  def setup
    skip "simple.onnx not found" unless model_exists?("simple.onnx")
  end

  def test_basic_model_integration
    model_path = simple_model_path

    klass = Class.new do
      include OnnxRuby::Model

      onnx_model model_path
      onnx_input ->(obj) { { "input" => [obj.features] } }
      onnx_output "output"

      attr_accessor :features

      def initialize(features)
        @features = features
      end
    end

    obj = klass.new([1.0, 2.0, 3.0, 4.0])
    result = obj.onnx_predict

    assert_kind_of Array, result
    assert_equal 1, result.length      # batch of 1
    assert_equal 3, result[0].length   # 3 output features
  end

  def test_model_uses_lazy_session
    model_path = simple_model_path

    klass = Class.new do
      include OnnxRuby::Model
      onnx_model model_path
      onnx_input ->(obj) { { "input" => [obj.data] } }

      attr_accessor :data
      def initialize(data) = @data = data
    end

    assert_kind_of OnnxRuby::LazySession, klass.onnx_session
    refute klass.onnx_session.loaded?

    obj = klass.new([1.0, 2.0, 3.0, 4.0])
    obj.onnx_predict

    assert klass.onnx_session.loaded?
  end

  def test_model_without_output_name
    model_path = simple_model_path

    klass = Class.new do
      include OnnxRuby::Model
      onnx_model model_path
      onnx_input ->(obj) { { "input" => [obj.data] } }

      attr_accessor :data
      def initialize(data) = @data = data
    end

    obj = klass.new([1.0, 2.0, 3.0, 4.0])
    result = obj.onnx_predict

    # Without onnx_output, returns first output's value
    assert_kind_of Array, result
  end

  def test_model_without_input_raises
    klass = Class.new do
      include OnnxRuby::Model
      onnx_model "anything.onnx"
    end

    obj = klass.new

    assert_raises(OnnxRuby::Error) do
      obj.onnx_predict
    end
  end

  def test_model_with_block_input
    model_path = simple_model_path

    klass = Class.new do
      include OnnxRuby::Model
      onnx_model model_path
      onnx_input { |obj| { "input" => [obj.data] } }

      attr_accessor :data
      def initialize(data) = @data = data
    end

    obj = klass.new([1.0, 2.0, 3.0, 4.0])
    result = obj.onnx_predict
    assert_kind_of Array, result
  end
end
