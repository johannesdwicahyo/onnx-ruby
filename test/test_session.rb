# frozen_string_literal: true

require "test_helper"

class TestSession < Minitest::Test
  include TestHelper

  def setup
    skip "simple.onnx not found — run: python script/create_test_models.py" unless model_exists?("simple.onnx")
    @session = OnnxRuby::Session.new(simple_model_path)
  end

  def test_load_model
    assert @session
  end

  def test_inputs_metadata
    inputs = @session.inputs
    assert_equal 1, inputs.length
    assert_equal "input", inputs[0][:name]
    assert_equal :float32, inputs[0][:type]
  end

  def test_outputs_metadata
    outputs = @session.outputs
    assert_equal 1, outputs.length
    assert_equal "output", outputs[0][:name]
    assert_equal :float32, outputs[0][:type]
  end

  def test_model_not_found
    assert_raises(OnnxRuby::ModelError) do
      OnnxRuby::Session.new("/nonexistent/model.onnx")
    end
  end
end
